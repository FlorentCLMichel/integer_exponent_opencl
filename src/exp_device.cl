kernel void exp_device (global unsigned int* output, 
                        global unsigned int* input,
                        global unsigned int n,
                        global unsigned int q)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t num_groups = get_num_groups(0);
    
    for (size_t i = 0u; i < num_groups; ++i) {
        size_t i = i * lsize + lid;
        unsigned int x = input[i];
        unsigned int m = 1;
        for (size_t j = 0u; j < n; ++j) {
            m = (m*x) % q;
        }
        output[i] = m;
    }
}
