kernel void exp_device (global unsigned int* output, 
                        global const unsigned int* input,
                        unsigned int n,
                        unsigned int q)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t num_groups = get_num_groups(0);
    
    for (size_t i = 0; i < num_groups; ++i) {
        size_t lidx = i * lsize + lid;
        unsigned int x = input[lidx];
        unsigned int m = 1;
        for (size_t j = 0; j < n; ++j) {
            m = (m*x) % q;
        }
        output[lidx] = m;
    }
}
