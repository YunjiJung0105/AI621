## Initials (5 points).
img = imread('data/chessboard_lightfield.png');

    lenslet = 16;
    [U, V] = deal(lenslet);
    S = size(img, 1) / lenslet;
    T = size(img, 2) / lenslet;
    C = 3;

    img_arr = zeros(U, V, S, T, C);
## Sub-aperture views (20 points).
    for s = 1:S
        for t = 1:T
            img_arr(:, :, s, t, :) = img((s-1)*lenslet+1:s*lenslet, (t-1)*lenslet+1:t*lenslet, :);
        end
    end

    img_mosaic = zeros(S*U, T*V, C);

    for refocused_u = 1:U
        for refocused_v = 1:V
            img_mosaic((refocused_u-1)*S+1:refocused_u*S, (refocused_v-1)*T+1:refocused_v*T, :) = img_arr(refocused_u, refocused_v, :, :, :);
        end
    end

    imwrite(uint8(img_mosaic), 'results/mosaic.png');



## Refocusing and focal-stack generation (40 points).
    maxUV = (lenslet - 1) / 2;
    refocused_u = (1:U) - 1 - maxUV;
    refocused_v = (1:V) - 1 - maxUV;
    focal_stack = zeros(8, S, T, C);  
    ind = 0;

    for range_d = 0:2:16
        ind = ind + 1;
        img_refocused = zeros(S, T, C);
        d = 0.1 * range_d;
        for u = 1:U
            for v = 1:V
                du = round(refocused_u(u) * d) * -1;
                dv = round(refocused_v(v) * d) * 1;
                img = squeeze(img_arr(u, v, :, :, :));

                img_shifted = zeros(size(img));
                shifted_idx = circshift(1:size(img, 1), du);
                img_shifted(shifted_idx, :, :) = img;
                shifted_idx = circshift(1:size(img, 2), dv);
                img_shifted(:, shifted_idx, :) = img_shifted;

                img_refocused = img_refocused + img_shifted / lenslet^2;    
            end
        end
        focal_stack(ind, :, :, :) = img_refocused;
        imwrite(uint8(img_refocused), strcat('results/refocused_d_', num2str(d), '.png'));
    end

    function shifted_idx = circshift(indices, shift_amount)
        num_elements = numel(indices);
        shift_amount = mod(shift_amount, num_elements);
        shifted_idx = [indices(shift_amount+1:end) indices(1:shift_amount)];
    end

## All-focus image and depth from defocus (35 points).
    sigma_1 = [0.5,1,1,2,4,6];
    sigma_2 = [0.5,1,2,2,4,6];

    img_all_focused = zeros(S, T, C);
    depth = zeros(S, T);
    w_sum = zeros(S, T);

    for sg_num = 1:6
        for d = 1:8
            img_rgb = squeeze(focal_stack(d, :, :, :));
            img_xyz = rgb2xyz(img_rgb);
            img_luminance = img_xyz(:, :, 2);
            img_low = imgaussfilt(img_luminance, sigma_1(sg_num));
            img_high = img_luminance - img_low;
            w_sharpness = imgaussfilt(img_high.^2, sigma_2(sg_num));

            img_all_focused = img_all_focused + bsxfun(@times, img_rgb, w_sharpness);
            depth = depth + w_sharpness * (d-1) * 0.2;
            w_sum = w_sum + w_sharpness;
        end

        img_all_focused = img_all_focused ./ w_sum;
        depth = depth ./ w_sum;

        imwrite(uint8(img_all_focused), strcat('results/all_focus_', num2str(sigma_1(sg_num)), '_', num2str(sigma_2(sg_num)), '.png'));
        imwrite(depth, strcat('results/depth_', num2str(sigma_1(sg_num)), '_', num2str(sigma_2(sg_num)), '.png'));
    end


