function[det,obj,ff] =missdetection(now_im,pre_im,pre_pos,pre_sz,non_compressed_features, compressed_features, w2c)
pos = [pre_pos(2) pre_pos(1)];
target_sz = [pre_sz(2) pre_sz(1)];
padding = 1;
sz = floor(target_sz * (1 + padding));
scale = 1;
alapha = 2.25;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
dist = rs.^2 + cs.^2;
conf = exp(-0.5 / (alapha) * sqrt(dist));
conf = conf/sum(sum(conf));
conff = fft2(conf);
hamming_window = hamming(sz(1)) * hann(sz(2))';
sigma = mean(target_sz);
window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));% use Hamming window to reduce frequency effect of image boundary
window = window/sum(sum(window));%normalization

img = pre_im;
if size(img,3) > 1,
    im = rgb2gray(img);
end
contextprior = get_context(im, pos, sz, window);
Hstcf = conff./(fft2(contextprior)+eps);

sigma = sigma*scale;% update scale in Eq.(15)
window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));%update weight function w_{sigma} in Eq.(11)
window = window/sum(sum(window));%normalization
%load image


img = now_im;
if size(img,3) > 1
    im = rgb2gray(img);
end
contextprior = get_context(im, pos, sz, window);%
response = real(ifft2(Hstcf.*fft2(contextprior))); %Eq.(11)
% [row, col] = find(response == max(response(:)), 1);
%  pos = pos - sz/2 + [row, col];
% target_sz([2,1]) = target_sz([2,1])*scale;% update object size
% rect_position = [pos([2,1]) - (target_sz([2,1])/2), (target_sz([2,1]))];
% imagesc(uint8(img))
% colormap(gray)
% rectangle('Position',rect_position,'LineWidth',4,'EdgeColor','r');

% padding = 1;
% num_compressed_dim = 2;
% lambda = 1e-2;
% output_sigma_factor = 1/16;
% sigma = 0.2;
%
% sz = floor(pre_sz * (1 + padding));
%
% % desired output (gaussian shaped), bandwidth proportional to target size
% output_sigma = sqrt(prod(pre_sz)) * output_sigma_factor;
% [rs, cs] = ndgrid((1:sz(2)) - floor(sz(2)/2), (1:sz(1)) - floor(sz(1)/2));
% y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
% yf = single(fft2(y));
%
% % store pre-computed cosine window
% cos_window = single(hann(sz(2)) * hann(sz(1))');
%
% [xo_npca, xo_pca] = get_subwindow(pre_im, [pre_pos(2),pre_pos(1)], [sz(2),sz(1)], non_compressed_features, compressed_features, w2c);
%  % initialize the appearance
%  z_npca = xo_npca;
%  z_pca = xo_pca;
%
%  % set number of compressed dimensions to maximum if too many
%  num_compressed_dim = min(num_compressed_dim, size(xo_pca, 2));
%
%   data_mean = mean(z_pca, 1);
%
%  % substract the mean from the appearance to get the data matrix
%  data_matrix = bsxfun(@minus, z_pca, data_mean);
%
%   % calculate the covariance matrix
%   cov_matrix = 1/(prod(sz) - 1) * (data_matrix' * data_matrix);
%
%   % calculate the principal components (pca_basis) and corresponding variances
%   [pca_basis, pca_variances, ~] = svd(cov_matrix);
%   projection_matrix = pca_basis(:, 1:num_compressed_dim);
%   projection_variances = pca_variances(1:num_compressed_dim, 1:num_compressed_dim);
%
%   old_cov_matrix = projection_matrix * projection_variances * projection_matrix';
%
%   [xo_npca, xo_pca] = get_subwindow(now_im, [pre_pos(2),pre_pos(1)], [sz(2),sz(1)], non_compressed_features, compressed_features, w2c);
%   x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);
%
%   kf = fft2(dense_gauss_kernel(sigma, x));
%   alphaf_num = yf .* kf;
%   alphaf_den = kf .* (kf + lambda);
%
%   zp = feature_projection(z_npca, z_pca, projection_matrix, cos_window);
%
% % extract the feature map of the local image patch
%  [xo_npca, xo_pca] = get_subwindow(now_im, [pre_pos(2),pre_pos(1)], [sz(2),sz(1)], non_compressed_features, compressed_features, w2c);
%
%   % do the dimensionality reduction and windowing
%    x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);
%
%   % calculate the response of the classifier
%    kf = fft2(dense_gauss_kernel(sigma, x, zp));
%    response = real(ifft2(alphaf_num .* kf ./ alphaf_den));

% target location is at the maximum response
if max(response(:)) > 0.004
    [row, col] = find(response == max(response(:)), 1);
    newpos = pre_pos - [pre_sz(1) pre_sz(2)] + [col, row];
    
    obj = [newpos(1) newpos(2) pre_sz(1) pre_sz(2)] ;
    det = [newpos(1)-1/2* pre_sz(1),newpos(2)-1/2* pre_sz(2),newpos(1)+1/2* pre_sz(1),newpos(2)+1/2* pre_sz(2),0.9];
    %   det = [newpos(1)-1/2* pre_sz(1),newpos(2)-1/2* pre_sz(2),newpos(1)+1/2* pre_sz(1),newpos(2)+1/2* pre_sz(2),max(response(:)*20];
    ff = 1;
    
else
    obj = [];
    det = [];
    ff = 0;
end
end