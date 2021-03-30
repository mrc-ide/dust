// -*- c++ -*-
#ifndef DUST_PREFIX_SCAN_CUH
#define DUST_PREFIX_SCAN_CUH

// See https://nvlabs.github.io/cub/classcub_1_1_block_scan.html
#ifdef __NVCC__

#include <cub/block/block_scan.cuh>

const int scan_block_size = 128;

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
namespace dust {

struct BlockPrefixCallbackOp
{
    // Running prefix
    int running_total;
    // Constructor
    DEVICE BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    DEVICE int operator()(int block_aggregate)
    {
        int old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

// cum_weights = cumsum(cum_weights)
template <typename real_t>
KERNEL void prefix_scan(real_t * cum_weights,
                        const size_t n_particles, const size_t n_pars) {
  const size_t n_particles_each = n_particles / n_pars;
  const int par_idx = blockIdx.x;
  // Specialize BlockScan for a 1D block of 128 threads
  typedef cub::BlockScan<int, scan_block_size> BlockScan;
  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;
  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items
  for (int block_offset = par_idx * n_particles_each;
       block_offset < (par_idx + 1) * n_particles_each;
       block_offset += scan_block_size) {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data = cum_weights[block_offset];
    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).InclusiveSum(
        thread_data, thread_data, prefix_op);
    cub::CTA_SYNC();
    // Store scanned items to output segment
    cum_weights[block_offset] = thread_data;
  }
}

}
#endif

#endif
