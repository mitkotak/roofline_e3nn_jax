import e3nn_jax
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

#RTX A5500
PEAK_FLOPS = 34.10/2
PEAK_BW = 0.768

BATCH = 100
LMAX = 7

WARMUP = 50
STEPS = 500

class TPe3nn(nn.Module):
    
    @nn.compact
    def __call__(self, x1, x2):
        # Does the einsum `bui,bvj,ijk,uvw->bwk` for every instruction

        irreps_out = e3nn_jax.tensor_product(x1.irreps, x2.irreps)
        tp = FunctionalFullyConnectedTensorProduct(x1.irreps, x2.irreps, irreps_out)
        w = [
            self.param(f"{i}", nn.initializers.normal(), i.path_shape)
            for i in tp.instructions
        ]
    
        return jax.vmap(tp.left_right, (None, 0, 0))(w, x1, x2)

from time import time
import numpy as np


def run(func, sample_func, *args):
    
    input_args = sample_func(*args)

    print("Warmup....")        
    for _ in range(WARMUP):
        result = func(*input_args)

    print("Benchmarking")
    timing = []
    for i in range(STEPS):
        input_args = sample_func(*args)
        start = time()
        result = func(*input_args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), result.array)
        end = time()
        if i > STEPS - 50:
            timing.append(end - start)

    return np.mean(timing)

def sample_e3nn_jax(tp_e3nn, irreps_in1, irreps_in2):
    rng = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(rng, 2)
    x = e3nn_jax.normal(irreps_in1, key1, (BATCH, ))
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), x.array)
    y = e3nn_jax.normal(irreps_in2, key2, (BATCH, ))
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), y.array)
    params = tp_e3nn.init(rng, x, y)
    return params, x, y


def main():
    roofline_gflops_list = []
    arithmetic_intensity_roofline_list = []

    fig, ax = plt.subplots()

    channels = [1, 32, 64, 128, 256]

    cm = mpl.colormaps['viridis'](np.linspace(0, 1, len(channels)))

    for i, channel in enumerate(channels):

        labels = []
        gflops_list = []
        arithmetic_intensity_list = []
        for lmax in range(1, LMAX+1):
        
            irreps_in1 = channel*e3nn_jax.s2_irreps(lmax)
            irreps_in2 = e3nn_jax.s2_irreps(lmax)
            print(f"{irreps_in1} \otimes {irreps_in2}")

            tp_e3nn = TPe3nn()
            tp_e3nn_apply = tp_e3nn.apply
            tp_e3nn_apply = jax.jit(tp_e3nn_apply)

            e3nn_jax_time = run(tp_e3nn_apply, sample_e3nn_jax, tp_e3nn, irreps_in1, irreps_in2)

            params, x_jax, y_jax = sample_e3nn_jax(tp_e3nn, irreps_in1, irreps_in2)
            nflops = nn.summary._get_flops(tp_e3nn_apply, params, x_jax, y_jax)
            _, weights_bytes = nn.summary._size_and_bytes(params)
            input_bytes = x_jax.array.nbytes + y_jax.array.nbytes
            result = tp_e3nn.apply(params, x_jax, y_jax)
            output_bytes = result.array.nbytes
            nbytes = input_bytes + output_bytes + weights_bytes

            arithmetic_intensity = nflops/nbytes

            roofline_gflops_per_sec = min(PEAK_FLOPS,
                                        arithmetic_intensity * PEAK_BW)

            gflops_per_sec = nflops*1e-12 / e3nn_jax_time

            print(f"e3nn-jax Arithmetic Intensity channels={channel}: {arithmetic_intensity}")
            arithmetic_intensity_list.append(arithmetic_intensity)
            arithmetic_intensity_roofline_list.append(arithmetic_intensity)
            print(f"e3nn-jax Roofline channels={channel}: {roofline_gflops_per_sec} GFLOPS/s")
            roofline_gflops_list.append(roofline_gflops_per_sec)
            print(f"e3nn-jax Bytes channels={channel}: {nbytes}")
            print(f"e3nn-jax channels={channel}: {gflops_per_sec} GFLOPS/s")
            gflops_list.append(gflops_per_sec)
        
        ax.scatter(arithmetic_intensity_list, gflops_list, label=f'embedding_{channel}', color=cm[i])
        
    ax.plot(arithmetic_intensity_roofline_list, roofline_gflops_list, '-', label='roofline', color='black')

    plt.title(f"EMBEDDING(1x0e + 1x1o + 1x2e + ....) \otimes 0e + 1o + 2e")
    plt.xlabel("Arithmetic Intensity (FLOPS/Bytes)")
    plt.grid(True)
    plt.ylabel("TFLOPS/s")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'TP_roofline.png')

if __name__ == "__main__":
    main()