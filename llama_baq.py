import time
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from gptq_baq import *
from modelutils import *
from quant_baq import *
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

# ---------------------------------------------------------------------------
# Small utility: return (result, elapsed_seconds)
# ---------------------------------------------------------------------------
def timed(fn, *args, **kwargs):
    start = time.time()
    ret = fn(*args, **kwargs)
    return ret, time.time() - start

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential_calib(model, dataloader, dev):
    print('Starting calibration for LLaMA...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            model(batch.to(dev))
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # Initialize containers for quantization
    ele_sum_container = {'value': 0.0}
    R_aver_container = {'value': 0.0}
    R_record_container = {'value': torch.zeros((len(layers), 6))}
    Gain_vec_container = {'value': []}
    loss_vec_container = {'value': []}
    
    quantizers = {}

    for i in tqdm(range(len(layers)), desc="Calibrating LLaMA layers"):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        # Skip these layers as they're not quantized
        subset = {name: subset[name] for name in subset if not name.endswith('_proj') and not 'rotary_emb' in name}
        
        name2idx = {name: idx for idx, name in enumerate(subset)}
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]
        
        for h in handles:
            h.remove()

        for name in subset:
            print(f"Quantizing layer {i}, {name}")
            gptq[name].fasterquant_calib(
                percdamp=args.percdamp, 
                groupsize=args.groupsize, 
                actorder=args.act_order, 
                static_groups=args.static_groups, 
                layer_idx=i, 
                name=name, 
                R_aver_container=R_aver_container, 
                ele_sum_container=ele_sum_container, 
                R_record_container=R_record_container, 
                col_idx=name2idx[name], 
                Gain_vec_container=Gain_vec_container, 
                loss_vec_container=loss_vec_container, 
                R_ref=args.wbits
            )
            quantizers[f'model.layers.{i}.{name}'] = gptq[name].quantizer
            gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if os.path.exists("gain_vec_tensor.pt"):
        os.remove("gain_vec_tensor.pt")
    gain_vec_tensor = torch.tensor(Gain_vec_container['value'])
    torch.save(gain_vec_tensor, "gain_vec_tensor.pt")

    if os.path.exists("loss_vec_tensor.pt"):
        os.remove("loss_vec_tensor.pt")
    loss_vec_tensor = loss_vec_container['value']
    torch.save(loss_vec_tensor, "loss_vec_tensor.pt")
    
    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting quantization for LLaMA...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            model(batch.to(dev))
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    ele_sum_container = {'value': 0.0}
    R_aver_container = {'value': 0.0}
    R_record_container = {'value': torch.zeros((len(layers), 6))}
    Gain_vec_container = {'value': []}

    filename_Ratio = "gain_vec_tensor.pt"
    Ratio_geo_ari = torch.load(filename_Ratio)
    Ratio_geo_ari = Ratio_geo_ari.view(len(layers), 6)

    filename_loss = "loss_vec_tensor.pt"
    loss_vec_tensor = torch.load(filename_loss)

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        # Skip these layers as they're not quantized
        subset = {name: subset[name] for name in subset if not name.endswith('_proj') and not 'rotary_emb' in name}
        
        name2idx = {name: idx for idx, name in enumerate(subset)}
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]
        
        for h in handles:
            h.remove()

        for name in subset:
            print(f"Quantizing layer {i}, {name}")
            gptq[name].fasterquant(
                percdamp=args.percdamp, 
                groupsize=args.groupsize, 
                actorder=args.act_order, 
                static_groups=args.static_groups, 
                layer_idx=i, 
                name=name, 
                R_aver_container=R_aver_container, 
                ele_sum_container=ele_sum_container, 
                R_record_container=R_record_container, 
                col_idx=name2idx[name], 
                Gain_vec_container=Gain_vec_container, 
                Ratio_geo_ari=Ratio_geo_ari, 
                loss_vec_tensor=loss_vec_tensor, 
                R_ref=args.wbits
            )
            quantizers[f'model.layers.{i}.{name}'] = gptq[name].quantizer
            gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    print(f"Final average rate = {R_aver_container['value']:.3f}")

    if os.path.exists("gain_vec_tensor.pt"):
        os.remove("gain_vec_tensor.pt")
    if os.path.exists("loss_vec_tensor.pt"):
        os.remove("loss_vec_tensor.pt")
    
    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating LLaMA...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(f"Processing layer {i}")
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                      attention_mask=attention_mask,
                      position_ids=position_ids)[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item()}")

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing LLaMA...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def load_quant3_llama(model, checkpoint):
    from transformers import LlamaConfig
    config = LlamaConfig.from_pretrained(model)
    
    def noop(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=args.faster_kernel)

    print('Loading LLaMA model...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model

def llama_multigpu(model, gpus):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            if 'position_ids' in kwargs:
                if cache['position_ids'] is None or cache['position_ids'].device != self.dev:
                    cache['position_ids'] = kwargs['position_ids'].to(self.dev)
                kwargs['position_ids'] = cache['position_ids']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LLaMA model to load; pass `meta-llama/Llama-2-7b-hf`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--base-eval', action='store_true',
        help='Evaluate and time the full‑precision model before quantisation.'
    )
    parser.add_argument(
        '--base-benchmark', type=int, default=0,
        help='If >0, run token‑by‑token latency benchmark on the full‑precision model (value = #tokens).'
    )

    args = parser.parse_args()

    if args.load:
        model = load_quant3_llama(args.model, args.load)
    else:
        model = get_llama(args.model)
        model.eval()

        if args.base_eval:
            print('=' * 60)
            print('💡  Evaluating FULL‑PRECISION LLaMA model (no quantisation)…')
            print('=' * 60)

            base_model = get_llama(args.model).eval().to(DEV)

            base_dl, base_testloader = get_loaders(
                args.dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=base_model.seqlen
            )

            _, secs = timed(llama_eval, base_model, base_testloader, DEV)
            print(f'⏱️  Full‑precision evaluation time: {secs:.1f} s')

            if args.base_benchmark > 0:
                toks = next(iter(base_dl))[0][:, :args.base_benchmark]
                benchmark(base_model, toks, check=args.check)

            base_model.cpu()
            torch.cuda.empty_cache()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers_calib = llama_sequential_calib(model, dataloader, DEV)
        model = get_llama(args.model)
        model.eval()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.load:
        exit()

    datasets = ['wikitext2']
    if args.new_eval:
        datasets = ['wikitext2', 'ptb', 'c4']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
