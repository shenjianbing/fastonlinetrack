  function response = patchmatch(pre_im,pre_pos,pre_sz,now_im,now_pos,now_sz,non_compressed_features, compressed_features, w2c)
         num_compressed_dim = 2;
         lambda = 1e-2;
         output_sigma_factor = 1/16;
         sigma = 0.2; 
%          fl = 0;
%          pos = zeros(1,2);
         
         [xo_npca, xo_pca] = get_subwindow(pre_im, [pre_pos(2),pre_pos(1)], [pre_sz(2),pre_sz(1)], non_compressed_features, compressed_features, w2c); 
         
         [x1,y1,x2,y2] = get_pregion(pre_im,pre_pos,pre_sz);
         [x11,y11,x22,y22] = get_region(now_im,now_pos,now_sz);
         
         if x11< x1 || y11<y1 || x22 > x2 || y22> y2
                response = 0;
         else
             im_patch =getpatch(now_im,now_pos,now_sz);
             im2 =mexResize(im_patch,[pre_sz(2),pre_sz(1)]);
             [z_npca,z_pca] = get_subwindow(im2, 1/2*[pre_sz(2),pre_sz(1)],[pre_sz(2),pre_sz(1)], non_compressed_features, compressed_features, w2c); 
             data_mean = mean(z_pca, 1);
             data_matrix = bsxfun(@minus, z_pca, data_mean);
             cov_matrix = 1/(prod(now_sz) - 1) * (data_matrix' * data_matrix);
             [pca_basis, ~, ~] = svd(cov_matrix);
             projection_matrix = pca_basis(:, 1:num_compressed_dim);
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
             response = res(floor(pre_sz(2)/2),floor(pre_sz(1)/2));
              if response * 1.2 < max(res(:)) %&& max(res(:))>0.8
                response = max(res(:));
%                 fl =1;
%                [m,n] =  find(res == max(res(:)), 1);
%                pos(2) = now_pos(2)-1/2*now_sz(2) + m/pre_sz(2)*now_sz(2);
%                pos(1) = now_pos(1)-1/2*now_sz(1) + n/pre_sz(1)*now_sz(1); 
% %                 now_sz = pre_sz;
             end
             overlap = com_overlap(now_pos,pre_pos,now_sz,pre_sz);
             response = (response + 0.5 * overlap)/1.5;
         end
end