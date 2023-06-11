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

<img width="642" alt="image" src="https://github.com/yoonjiJung/AI621/assets/75105873/86a5f59d-8269-4709-b20c-ada13411852a">


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

![refocused_d_0](https://github.com/yoonjiJung/AI621/assets/75105873/0d04cf62-361c-4318-8449-cdd03767d853)

![refocused_d_0 2](https://github.com/yoonjiJung/AI621/assets/75105873/5c8ebbd4-b551-4f76-b99b-2870d3902b6f)

![refocused_d_0 4](https://github.com/yoonjiJung/AI621/assets/75105873/83abc28c-23e5-45da-a196-65755f1ca11e)

![refocused_d_0 6](https://github.com/yoonjiJung/AI621/assets/75105873/ecb609a1-fd44-45a4-b419-6f830d8c0927)

![refocused_d_0 8](https://github.com/yoonjiJung/AI621/assets/75105873/82af5f5c-3a0e-42ea-8b4e-41ebf52710c9)

![refocused_d_1](https://github.com/yoonjiJung/AI621/assets/75105873/6c278762-386c-4866-bd6b-f008b5a524d4)

![refocused_d_1 2](https://github.com/yoonjiJung/AI621/assets/75105873/8d56c60d-4d7f-4355-86b9-52c274010078)

![refocused_d_1 4](https://github.com/yoonjiJung/AI621/assets/75105873/c211ddc5-24c1-4306-a6ed-5b675ed12c35)


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

![all_focus_0 5_0 5](https://github.com/yoonjiJung/AI621/assets/75105873/8076a1da-02f5-4540-bcae-fa96cd1f7bbc)

![depth_0 5_0 5](https://github.com/yoonjiJung/AI621/assets/75105873/ae6cdd55-80c8-4773-b200-e8e83f2037f2)

![all_focus_1_1](https://github.com/yoonjiJung/AI621/assets/75105873/d11da0da-9cb8-418a-a548-936d35c21aa8)

![depth_1_1](https://github.com/yoonjiJung/AI621/assets/75105873/2076cc67-2245-4110-9fcb-e11095209363)

![all_focus_1_2](https://github.com/yoonjiJung/AI621/assets/75105873/cb71b525-c87a-4e26-87cb-8f16b3409017)

![depth_1_2](https://github.com/yoonjiJung/AI621/assets/75105873/29a1478f-9b2e-4564-be01-05036fd5a74b)

![all_focus_2_2](https://github.com/yoonjiJung/AI621/assets/75105873/86fa16c0-f766-4e14-9c00-4c530cd94139)

![depth_2_2](https://github.com/yoonjiJung/AI621/assets/75105873/f092ec7a-3ee0-4401-9885-38857a69c7b8)

![all_focus_4_4](https://github.com/yoonjiJung/AI621/assets/75105873/5b12e65e-66f8-4581-a53e-5195a6ef26f5)

![depth_4_4](https://github.com/yoonjiJung/AI621/assets/75105873/fa2de9a2-ef46-49bc-befe-c22848265dee)

![all_focus_6_6](https://github.com/yoonjiJung/AI621/assets/75105873/89057c52-278a-430b-91ca-78f1c318656d)

![depth_6_6](https://github.com/yoonjiJung/AI621/assets/75105873/d3296d2f-89da-4f04-b06a-d520c5950302)
