## 1. LINEARIZE RENDERED IMAGES (25 POINTS)
* I implemented g function and least squares to linearize the rendered images. I plotted the g function for each weight scheme, uniform and tent.

        %% Least square
        logt = log(T);

        [h, w, c, p] = size(img);

        patch = reshape(img, h*w, c, p);
        disp(size(patch));

        z_max = 252;
        z_min = 3;
        z_mid = (z_max + z_min) * 0.5;

        WEIGHT = 'uniform';
        weight_ = zeros(256,1);

        if strcmp(WEIGHT, 'uniform')
            weight_(:) = 1;
        elseif strcmp(WEIGHT, 'tent')
            for i = (z_min+1):(z_max+1)
                if (i-1) > z_mid
                    weight_(i) = z_max - (i - 1);
                else
                    weight_(i) = (i - 1) - z_min;
                end
            end
        end

        weight_ = double(weight_) / max(weight_);
        weight_ = (weight_ + 0.01)*0.99;

        [g, logE] = leastSquare(uint16(patch), logt, 10, weight_);

        %% Normalize g
        clearvars g_exp

        g_ = g;
        g_exp = exp(g_);

        g_expmin = min(g_exp, [], 1);

        for i = 1:3
            g_exp(1:z_max-2, i) = g_exp(1:z_max-2, i) - g_expmin(i);
        end

        g_expmax = max(g_exp, [], 1);

        for i = 1:3
            g_exp(1:z_max-2, i) = g_exp(1:z_max-2, i)/g_expmax(i);
        end

        plot(g_exp);

        %% Linearization
        clearvars I_lin reimg g_ref;
        reimg = imresize(img_org, 0.2);
        reimg(reimg>252)=252;
        reimg(reimg<3)=3;
        [rh, rw, rc, rp] = size(reimg);
        I_lin = zeros(400, 600, 3, 16);

        for p_=1:rp
            for c_=1:rc
                g_ref = g_exp(:, c);
                for h_=1:rh
                    for w_=1:rw
                        I_lin(h_, w_, c_, p_) = g_ref(uint8(reimg(h_, w_, c_, p_) -2));
                    end
                end
            end
        end


        function [g, logE] = leastSquare(img, logt, lambda, weight)
        [N, c, p] = size(img);
        disp(size(img));
        z_max = 252;
        z_min = 3;
        z_range = z_max - z_min + 1;

        disp('Alloc matrix');
        A = spalloc(N*p+z_range+1, z_range+N, N*p*2 + z_range*3);
        b = zeros(size(A, 1), 1);

        for c_=1:c
            tic
            disp('Construct A and b');
            disp(c_);
            k_ = 1;
            for i=1:N
                for j=1:p
                    intensity = img(i, c_, j);
                    wght = weight(intensity + 1);
                    A(k_, intensity + 1) = wght;
                    A(k_, i+z_range) = -wght;
                    b(k_, 1) = wght * logt(j);
                    k_ = k_ + 1;
                end
            end

            A(k_, z_max) = 1;

            k_ = k_ + 1;

            for i_=1:(z_max-2)
                A(k_, i_) = lambda * weight(i_+1);
                A(k_, i_+1) = -2*lambda * weight(i_+1);
                A(k_, i_+2) = lambda * weight(i_+1);
                k_ = k_ + 1;
            end
            toc

            tic
            disp('Find pseudo inverse');
            tmp = A\b;
            disp(size(tmp));
            g(:, c_) = tmp(1:z_range);
            logE(:, c_) = tmp(z_range+1:size(tmp,1));
            toc
        end
        end


* Uniform plot
![uniform_gexp](https://github.com/yoonjiJung/AI621/assets/75105873/ed42ea36-fd7f-44e8-8f20-5f7809bdfdda)

* Tent plot
![tent](https://github.com/yoonjiJung/AI621/assets/75105873/6e74794a-39b4-4c08-b554-4fa3ddec4929)


## 2. MERGE EXPOSURE STACK INTO HDR IMAGE (15 POINTS)
* I implemented merging skills and merged the rendered images into 4 ways combining (uniform / tent) and (linear / log).

        MERGE = 'log';

        linmin = min(I_lin,[], [1,2,4]);
        linmax = max(I_lin,[], [1,2,4]);

        for i = 1:3
            tmptttt(:, :, i, :) = I_lin(:, :, i, :) - linmin(i);
            tmptttt(:, :, i, :) = double(I_lin(:, :, i, :))/double(linmax(i));
        end

        tmptttt = uint8(round(tmptttt*252));

        doubleimg = double(reimg)/252.0;
        I_HDR = merge(tmptttt, doubleimg, weight_, T, MERGE);

        hdrwrite(I_HDR, 'HDRFILE.hdr');
        HDRFILE = hdrread('HDRFILE.hdr');

        rgb = tonemap(HDRFILE);
        rrr=rgb(:, :, 1);
        ggg=rgb(:, :, 2);
        bbb=rgb(:, :, 3);

        figure;
        subplot(4,1,1),imshow(rrr, []);
        subplot(4,1,2),imshow(ggg, []);
        subplot(4,1,3),imshow(bbb, []);
        subplot(4,1,4),imshow(rgb);


        function HDR = merge(img_lin, img_org, weight, T, MERGE)

        [h, w, c, p] = size(img_lin);

        HDR = zeros(h, w, c);

        numer = zeros(h,w,c);
        denom = zeros(h,w,c);

        for p_=1:p
            for c_=1:c
                for h_=1:h
                    for w_=1:w
                        tmp_lin = img_lin(h_, w_, c_, p_);
                        tmp_org = img_org(h_, w_, c_, p_);
                        if strcmp(MERGE, 'linear')
                            numer(h_, w_, c_) = numer(h_, w_, c_) + weight(tmp_lin + 1) * tmp_org / T(p_); 
                        else
                            numer(h_, w_, c_) = numer(h_, w_, c_) + weight(tmp_lin + 1) * (log(double(tmp_org)) - log(T(p_))); 
                        end
                        denom(h_, w_, c_) = denom(h_, w_, c_) + weight(tmp_lin + 1);   
                    end
                end
            end
            if denom(h_, w_, c_)==0
                numer(h_, w_, c_)=1;
                denom(h_, w_, c_)=1;
            end
        end

        if strcmp(MERGE, 'linear')
            HDR = numer./denom; 
        else
            HDR = exp(numer./denom);  
        end
        end


* Uniform / Linear
![uniform_linear](https://github.com/yoonjiJung/AI621/assets/75105873/8ff57b9a-6542-42c4-941f-23db0ed937ae)

* Uniform / Log
![uniform_log](https://github.com/yoonjiJung/AI621/assets/75105873/74b5fa3e-8268-494f-afd6-f718d0cc6ca2)

* Tent / Linear
![tent_linear](https://github.com/yoonjiJung/AI621/assets/75105873/e0cb959b-88c5-4541-9276-2a9904d15130)

* Tent / Log
![tent_log](https://github.com/yoonjiJung/AI621/assets/75105873/0e1b4e1a-738b-4a8f-86ff-0094a8246fb8)


## 3. EVALUATION (10 POINTS)
* Via color checker, I determined each patch position. Then, for the average luminance, I fitted linear regression and checked RMSE and R-squared values for each way. It is shown that RMSE is the smallest for uniform / linear and R-squared is the highest for uniform / log.

        patch = [375 62 387 73; 376 78 388 89; 376 93 388 105; 377 110 388 122; 377 125 389 138; 378 141 390 153;];

        figure;
        idx = 1;
        weights = {'uniform', 'tent'};
        merges = {'linear', 'log'};

        for weight_num = 1:2
            for merge_num = 1:2
                hdr_path = sprintf('%s_%s.hdr', weights{weight_num}, merges{merge_num});
                img = hdrread(hdr_path);
                img_XYZ = rgb2xyz(img, 'Colorspace', 'linear-rgb');

                Lum = img_XYZ(:,:,2);

                x = zeros(6, 1);
                y = zeros(6, 1);

                for i = 1:6  
                    x(i) = i;
                    y(i) = log(mean(mean(Lum(patch(i,2):patch(i,4), patch(i,1):patch(i,3)))));
                end

                fit = fitlm(x, y);
                fit.RMSE
                fit.Rsquared

                subplot(4,1,idx);
                plot(fit)
                title(sprintf('%s_%s', weights{weight_num}, merges{merge_num}))

                idx = idx+1;
            end
        end

* Results
![eval](https://github.com/yoonjiJung/AI621/assets/75105873/e80c0cbd-3bbd-4a1b-ae1a-ffccbcd5bb54)

* RMSE

uniform / linear : 0.0026

uniform / log : 0.0095

tent / linear : 0.0273

tent / log : 0.0337

* R-spuared

uniform / linear : 0.8891

uniform / log : 0.9991

tent / linear : 0.9819

tent / log :  0.9959

## 4. PHOTOGRAPHIC TONEMAPPING (20 POINTS)
* For tonemapping, I chose uniform / log img on the above because it has the highest R-squared and I think it is the best when I look at it. I applied photographic tonemapping in 2 ways : RGB and Luminance. I changed parameters. I prefer applying the photographic tonemapping separately for each RGB channel because it looks more natural. 

        function [ I_c ] = photographic_tonemap_func( I, K, B, e )
        I_m = exp(mean(mean(log(I + e))));
        I_ = (K / I_m) * I;
        I_white = B * max(max(I_));
        I_c = I_ .* (1 + I_ / (I_white * I_white)) ./ (1 + I_);
        end

        function [ output ] = photographic_tonemapping( img, K, B, colorSpace )
        N = size(img, 1) * size(img, 2);
        e = 1e-6;
        [h, w, c] = size(img);

        if strcmp(colorSpace, 'rgb')
            output = zeros(h, w, c);

            for c = 1:c
                I = img(:,:,c);
                I_c = photographic_tonemap_func(I, K, B, e);
                output(:,:,c) = I_c;
            end

        else
            img_xyz = rgb2xyz(img, 'Colorspace', 'linear-rgb');
            X = img_xyz(:,:,1);
            Y = img_xyz(:,:,2);
            Z = img_xyz(:,:,3);

            x = X ./ (X + Y + Z);
            y = Y ./ (X + Y + Z);

            I_c = photographic_tonemap_func(Y, K, B, e);
            [X, Y, Z] = xyY_to_XYZ(x, y, I_c);

            img_xyz_ = zeros(h, w, c);
            img_xyz_(:,:,1) = X;
            img_xyz_(:,:,2) = Y;
            img_xyz_(:,:,3) = Z;

            output = xyz2rgb(img_xyz_);
        end
        end

* RGB 

K = 0.7 / B = 0.95
![photo_rgb_K07_B095](https://github.com/yoonjiJung/AI621/assets/75105873/6060451b-f5fe-4866-8ebf-a76ff91b3676)

K = 0.2 / B = 0.95
![photo_rgb_K02_B095](https://github.com/yoonjiJung/AI621/assets/75105873/42d0b832-28d3-4b42-9f4a-c7941d47c70b)

* Luminance

K = 0.15 / B = 0.95
![photo_xyY_K015_B095](https://github.com/yoonjiJung/AI621/assets/75105873/eaf4eef6-8601-4da7-8a6b-2a0e4b6f9dd9)

K = 0.5 / B = 0.95
![photo_xyY_K05_B095](https://github.com/yoonjiJung/AI621/assets/75105873/66bc4590-6466-4ae2-bc7f-42c1af906d42)


## 5. TONEMAPPING USING BILATERAL FILTERING (30 POINTS)
* I also chose uniform / log img image for tonemapping. I applied bilateral filtering tonemapping in 2 ways : RGB and Luminance. I changed parameters, but there was no big difference. I prefer applying the bilateral filtering tonemapping separately for each RGB channel because the edges are more accurate. 

        function [ I_tmp ] = bilateral_tonemap_func(I, S, degree, sigma, e)
        L = log(I + e);
        L_min = min(min(L));
        L_max = max(max(L));

        L_temp = (L - L_min) / (L_max - L_min);
        B_temp = imbilatfilt(L_temp, degree, sigma);
        B = B_temp * (L_max - L_min) + L_min;
        D = L - B;
        B_new = S * (B - max(max(B)));
        I_tmp = exp(B_new + D);
        end


        function [ output ] = bilateral_tonemapping( img, degree, S, sigma, colorSpace )
        N = size(img, 1) * size(img, 2);
        e = 1e-6;
        [h, w, c] = size(img);

        if strcmp(colorSpace, 'rgb')
            output = zeros(h, w, c);

            for c = 1:c
                I_tmp = bilateral_tonemap_func(img(:,:,c), S, degree, sigma, e);
                output(:,:,c) = I_tmp;
            end

        else
            img_xyz = rgb2xyz(img, 'Colorspace', 'linear-rgb');

            X = img_xyz(:,:,1);
            Y = img_xyz(:,:,2);
            Z = img_xyz(:,:,3);

            x = X ./ (X + Y + Z);
            y = Y ./ (X + Y + Z);

            I_tmp = bilateral_tonemap_func(Y, S, degree, sigma, e);        
            [X, Y, Z] = xyY_to_XYZ(x, y, I_tmp);

            img_xyz_ = zeros(h, w, c);
            img_xyz_(:,:,1) = X;
            img_xyz_(:,:,2) = Y;
            img_xyz_(:,:,3) = Z;

            output = xyz2rgb(img_xyz_);
        end
        end

* RGB - degree = 50 / sigma = 7
![bilateral_degree50_sigma7](https://github.com/yoonjiJung/AI621/assets/75105873/46705917-ebde-4c3d-922b-3a5452e3f6bf)

* Luminance - degree = 50 / sigma = 7
![bilateral_xyY_degree50_sigma7](https://github.com/yoonjiJung/AI621/assets/75105873/f59863e9-4756-4e31-aab1-6e9e86f050ad)

