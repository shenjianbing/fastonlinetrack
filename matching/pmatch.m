function virscore = pmatch(vir_im,virobjects,now_im,now_pos,now_sz,non_compressed_features, compressed_features, w2c)
         
         num_compressed_dim = 2;
         lambda = 1e-2;
         output_sigma_factor = 1/16;
         sigma = 0.2; 
         
         pre_pos = virobjects(1,1:2);
         pre_sz = virobjects(1,3:4);
         [xo_npca, xo_pca] = get_subwindow(vir_im, [pre_pos(2),pre_pos(1)], [pre_sz(2),pre_sz(1)], non_compressed_features, compressed_features, w2c);    
         im_patch =getpatch(now_im,now_pos,now_sz);
         im2 =mexResize(im_patch,[pre_sz(2),pre_sz(1)]);
         [z_npca,z_pca] = get_subwindow(im2, 1/2*[pre_sz(2),pre_sz(1)],[pre_sz(2),pre_sz(1)], non_compressed_features, compressed_features, w2c); 
         data_mean = mean(z_pca, 1);
         data_matrix = bsxfun(@minus, z_pca, data_mean);
         cov_matrix = 1/(prod(now_sz) - 1) * (data_matrix' * data_matrix);
         [pca_basis, pca_variances, ~] = svd(cov_matrix);
          projection_matrix = pca_basis(:, 1:num_compressed_dim);
          projection_variances = pca_variances(1:num_compressed_dim, 1:num_compressed_dim);
          cos_window = single(hann(pre_sz(2)) * hann(pre_sz(1))');
          zp = feature_projection(z_npca, z_pca, projection_matrix, cos_window);
          x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);
          kf = fft2(dense_gauss_kernel(sigma, zp));
          output_sigma = sqrt(prod(pre_sz)) * output_sigma_factor;
          [rs, cs] = ndgrid(((1:pre_sz(2)) - floor(pre_sz(2))/2), (1:pre_sz(1)) - floor(pre_sz(1)/2));
           y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
           yf = single(fft2(y));
           alphaf_num = yf .* kf;
           alphaf_den = kf .* (kf + lambda);
           kf = fft2(dense_gauss_kernel(sigma, x, zp));
           res = real(ifft2(alphaf_num .* kf ./ alphaf_den));
           virscore = res(floor(pre_sz(2)/2),floor(pre_sz(1)/2));
         
end