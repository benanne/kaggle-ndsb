import numpy as np
import theano
import theano.sandbox.cuda as cuda

from pycuda.compiler import SourceModule

import theano.misc.pycuda_init



class PyCudaOp(cuda.GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        raise NotImplementedError

    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])


class CyclicRollOp(PyCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        mod = SourceModule("""
            __global__ void cyclic_roll(float * input, float * output, int batch_size, int num_features) {
                int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
                int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim

                int height = 4 * batch_size;
                int width = 4 * num_features;

                if (x < num_features && y < height) {
                    for (int i = 0; i < 4; i++) {
                        int y_out = (y + batch_size * (4 - i)) % height;
                        int x_out = x + num_features * i;

                        output[y_out * width + x_out] = input[y * num_features + x];
                    }
                }
            }""")
        kernel = mod.get_function("cyclic_roll")

        def thunk():
            in_shape = inputs[0][0].shape
            rows, cols = in_shape

            assert rows % 4 == 0

            out_shape = (rows, 4 * cols)
            
            batch_size = rows // 4
            num_features = cols

            out = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if out[0] is None or out[0].shape != out_shape:
                out[0] = cuda.CudaNdarray.zeros(out_shape)

            x_block = 16
            y_block = 16
            block = (x_block, y_block, 1)

            x_grid = int(np.ceil(float(in_shape[1]) / x_block))
            y_grid = int(np.ceil(float(in_shape[0]) / y_block))
            grid = (x_grid, y_grid, 1)

            kernel(inputs[0][0], out[0], np.intc(batch_size), np.intc(num_features), block=block, grid=grid)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inp, grads):
        top, = grads
        top = cuda.basic_ops.gpu_contiguous(top)
        return [CyclicRollGradOp()(top)]


cyclic_roll = CyclicRollOp()


class CyclicRollGradOp(PyCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        mod = SourceModule("""
            __global__ void cyclic_roll_grad(float * input, float * output, int batch_size, int num_features) {
                int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
                int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim

                int height = 4 * batch_size;
                int width = 4 * num_features;

                float val = 0;

                if (x < num_features && y < height) {
                    for (int i = 0; i < 4; i++) {
                        int y_in = (y + batch_size * (4 - i)) % height;
                        int x_in = x + num_features * i;

                        val += input[y_in * width + x_in];
                    }

                    output[y * num_features + x] = val;
                }
            }""")
        kernel = mod.get_function("cyclic_roll_grad")

        def thunk():
            in_shape = inputs[0][0].shape
            rows, cols = in_shape
            
            assert rows % 4 == 0
            assert cols % 4 == 0

            out_shape = (rows, cols // 4)
            
            batch_size = rows // 4
            num_features = cols // 4

            out = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if out[0] is None or out[0].shape != out_shape:
                out[0] = cuda.CudaNdarray.zeros(out_shape)

            x_block = 16
            y_block = 16
            block = (x_block, y_block, 1)

            x_grid = int(np.ceil(float(out_shape[1]) / x_block))
            y_grid = int(np.ceil(float(out_shape[0]) / y_block))
            grid = (x_grid, y_grid, 1)

            kernel(inputs[0][0], out[0], np.intc(batch_size), np.intc(num_features), block=block, grid=grid)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


class CyclicConvRollOp(PyCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        mod = SourceModule("""
            __global__ void cyclic_convroll(float * input, float * output, int batch_size, int num_channels, int map_size) {
                int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
                int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim

                int map_size_sq = map_size * map_size;
                int example_size = num_channels * map_size_sq;
                int num_rows = 4 * batch_size; // number of rows in the input/output, seen as a 2D array
                int num_cols = 4 * example_size; // number of columns in the output, seen as a 2D array

                // feature indices (channels, height, width)
                int x_channel = x / map_size_sq;
                int x_f0 = (x % map_size_sq) / map_size;
                int x_f1 = x % map_size;

                int x_out_f0 = x_f0;
                int x_out_f1 = x_f1;
                int tmp;

                if (x < example_size && y < num_rows) {
                    for (int i = 0; i < 4; i++) {
                        int y_out = (y + batch_size * (4 - i)) % num_rows;
                        int x_out = example_size * i + x_channel * map_size_sq + x_out_f0 * map_size + x_out_f1;

                        output[y_out * num_cols + x_out] = input[y * example_size + x];
                        // note that the writes to output go in reverse order for all the rotated feature maps.
                        // this may slow things down a little, perhaps there is room for further optimization.

                        // rotate
                        tmp = x_out_f0;
                        x_out_f0 = x_out_f1;
                        x_out_f1 = map_size - 1 - tmp;
                    }
                }
            }""")
        kernel = mod.get_function("cyclic_convroll")

        def thunk():
            in_shape = inputs[0][0].shape
            full_batch_size, num_channels, height, width = in_shape
            assert height == width  # else convroll doesn't make sense
            assert full_batch_size % 4 == 0

            out_shape = (full_batch_size, 4 * num_channels, height, width)
            
            batch_size = full_batch_size // 4
            example_size = num_channels * height * width
            map_size = height

            out = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if out[0] is None or out[0].shape != out_shape:
                out[0] = cuda.CudaNdarray.zeros(out_shape)

            x_block = 16
            y_block = 16
            block = (x_block, y_block, 1)

            x_grid = int(np.ceil(float(example_size) / x_block))
            y_grid = int(np.ceil(float(full_batch_size) / y_block))
            grid = (x_grid, y_grid, 1)

            kernel(inputs[0][0], out[0], np.intc(batch_size), np.intc(num_channels), np.intc(map_size), block=block, grid=grid)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inp, grads):
        top, = grads
        top = cuda.basic_ops.gpu_contiguous(top)
        return [CyclicConvRollGradOp()(top)]


cyclic_convroll = CyclicConvRollOp()


class CyclicConvRollGradOp(PyCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        mod = SourceModule("""
            __global__ void cyclic_convroll_grad(float * input, float * output, int batch_size, int num_channels, int map_size) {
                int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
                int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim

                int map_size_sq = map_size * map_size;
                int example_size = num_channels * map_size_sq;
                int num_rows = 4 * batch_size; // number of rows in the input/output, seen as a 2D array
                int num_cols = 4 * example_size; // number of columns in the input, seen as a 2D array

                // feature indices (channels, height, width)
                int x_channel = x / map_size_sq;
                int x_f0 = (x % map_size_sq) / map_size;
                int x_f1 = x % map_size;

                int x_in_f0 = x_f0;
                int x_in_f1 = x_f1;
                int tmp;

                float val;

                if (x < example_size && y < num_rows) {
                    for (int i = 0; i < 4; i++) {
                        int y_in = (y + batch_size * (4 - i)) % num_rows;
                        int x_in = example_size * i + x_channel * map_size_sq + x_in_f0 * map_size + x_in_f1;

                        val += input[y_in * num_cols + x_in];

                        // rotate
                        tmp = x_in_f0;
                        x_in_f0 = x_in_f1;
                        x_in_f1 = map_size - 1 - tmp;
                    }

                    output[y * example_size + x] = val;
                }
            }""")
        kernel = mod.get_function("cyclic_convroll_grad")

        def thunk():
            in_shape = inputs[0][0].shape
            full_batch_size, num_channels_rolled, height, width = in_shape
            assert height == width  # else convroll doesn't make sense
            assert full_batch_size % 4 == 0
            assert num_channels_rolled % 4 == 0

            num_channels = num_channels_rolled // 4
            batch_size = full_batch_size // 4
            out_shape = (full_batch_size, num_channels, height, width)
            
            example_size = num_channels * height * width
            map_size = height

            out = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if out[0] is None or out[0].shape != out_shape:
                out[0] = cuda.CudaNdarray.zeros(out_shape)

            x_block = 16
            y_block = 16
            block = (x_block, y_block, 1)

            x_grid = int(np.ceil(float(example_size) / x_block))
            y_grid = int(np.ceil(float(full_batch_size) / y_block))
            grid = (x_grid, y_grid, 1)

            kernel(inputs[0][0], out[0], np.intc(batch_size), np.intc(num_channels), np.intc(map_size), block=block, grid=grid)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk
