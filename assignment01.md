# AI621 HW\#01

## INITIALS (5 PTS)
    img = imread('./data/banana_slug.tiff');  
    img_size = size(img); 
    fprintf("### Initials (read images and get metadata)\n");
    fprintf("Datasize: %d %d\n", img_size(1), img_size(2));
    fprintf("Datatype before typecasting: %s\n", class(img));
    img = cast(img, 'double');
    fprintf("Datatype after typecasting: %s\n", class(img));

* results
    Datasize: 2856 4290
    Datatype before typecasting: uint16
    Datatype after typecasting: double


## LINEARIZATION (5 POINTS)
    fprintf("### Linearization\n");
    M = max(img,[],[1,2]);
    m = min(img,[],[1,2]);
    fprintf("Before linearization / Max: %f \t Min: %f\n", M, m);
    img = (img - 2047); % remove zeros
    img(img<0) = 0; 
    img(img>15000) = 0; % remove over-exposed
    M = max(img,[],[1,2]);  % 15303 / min : 2002
    img = img/M;

    M = max(img,[],[1,2]);  % max to 1
    m = min(img,[],[1,2]);  % min to 0 

    fprintf("After linearization / Max: %f \t Min: %f\n", M, m);

* results
Before linearization / Max: 15303.000000 	 Min: 2002.000000
After linearization / Max: 1.000000 	 Min: 0.000000


## IDENTIFYING THE CORRECT BAYER PATTERN (20 POINTS)
    figure;
    % Pattern if grbg
    im1 = img(1:2:end, 2:2:end);
    im2 = img(1:2:end, 1:2:end);
    im3 = img(2:2:end, 1:2:end);
    im_rgb = cat(3, im1/ max(im1,[],[1,2]), im2/ max(im2,[],[1,2]), im3/ max(im3,[],[1,2]));
    subplot(1, 4, 1);imshow(min(1, im_rgb * 5));title('GRBG');

    % Pattern if rggb
    im1 = img(1:2:end, 1:2:end);
    im2 = img(2:2:end, 1:2:end);
    im3 = img(2:2:end, 2:2:end);
    im_rgb = cat(3, im1/ max(im1,[],[1,2]), im2/ max(im2,[],[1,2]), im3/ max(im3,[],[1,2]));
    subplot(1, 4, 2);imshow(min(1, im_rgb * 5));title('RGGB');

    % Pattern if bggr
    im1 = img(2:2:end, 2:2:end);
    im2 = img(1:2:end, 2:2:end);
    im3 = img(1:2:end, 1:2:end);
    im_rgb = cat(3, im1/ max(im1,[],[1,2]), im2/ max(im2,[],[1,2]), im3/ max(im3,[],[1,2]));
    subplot(1, 4, 3);imshow(min(1, im_rgb * 5));title('BGGR');

    % Pattern if gbrg
    im1 = img(2:2:end, 1:2:end);
    im2 = img(1:2:end, 1:2:end);
    im3 = img(1:2:end, 2:2:end);
    im_rgb = cat(3, im1/ max(im1,[],[1,2]), im2/ max(im2,[],[1,2]), im3/ max(im3,[],[1,2]));
    subplot(1, 4, 4);imshow(min(1, im_rgb * 5));title('GBRG');

    disp('Correct Bayer Pattern: RGGB');

    disp("Verifying by inspecting the pixel values");
    % Dominant color of top left of the img : green
    tmp = img(1:1:4, 1:1:4);
    disp(tmp);
    % Dominant color of bottom left of the img : blue
    tmp = img(2852:1:end, 1:1:4);
    disp(tmp);

* results
<img width="1152" alt="image" src="https://user-images.githubusercontent.com/75105873/227754679-52ed5284-78e6-497b-883b-0b83e53c95f7.png">

Verifying by inspecting the pixel values
    0.0128    0.0260    0.0137    0.0309
    0.0272    0.0122    0.0317    0.0143
    0.0145    0.0294    0.0149    0.0327
    0.0273    0.0133    0.0306    0.0123

    0.0066    0.0065    0.0091    0.0068
    0.0020    0.0063    0.0018    0.0056
    0.0064    0.0075    0.0056    0.0072
    0.0032    0.0045    0.0028    0.0057
    0.0059    0.0063    0.0051    0.0045
    

## WHITE BALANCING (20 POINTS)
    % Set pattern as 'rggb'
    R = img(1:2:end, 1:2:end); 
    G1 = img(1:2:end, 2:2:end); 
    G2 = img(2:2:end, 1:2:end); 
    G = (G1 + G2)/2;
    B = img(2:2:end, 2:2:end); 
    img_rgb = cat(3, R, G, B);

    % White world assumption
    img_ww = img_rgb;
    ch_max = max(img_ww, [], [1,2]);

    img_ww(:,:,1) = img_ww(:,:,1).* (ch_max(2)/ch_max(1));
    img_ww(:,:,3) = img_ww(:,:,3).* (ch_max(2)/ch_max(3));
    figure,subplot(1,2,1),imshow(min(img_ww, 1)),title('WhiteWorld');

    % Gray world assumption
    img_gw = img_rgb;
    ch_mean = mean(img_rgb, [1,2]);

    img_gw(:,:,1) = img_gw(:,:,1)* (ch_mean(2)/ch_mean(1));
    img_gw(:,:,3) = img_gw(:,:,3)* (ch_mean(2)/ch_mean(3));
    subplot(1,2,2),imshow(min(img_gw, 1)),title('GrayWorld');

* results
<img width="1150" alt="image" src="https://user-images.githubusercontent.com/75105873/227754757-747ea69d-8b60-40ec-b6c1-f0c52bd1eeaf.png">


## DEMOSAICING (25 POINTS)
    % White
    r_white = interp2(img_ww(:,:,1),1);
    g_white = interp2(img_ww(:,:,2),1);
    b_white = interp2(img_ww(:,:,3),1);
    img_white_dm = cat(3, r_white, g_white, b_white);
    subplot(1,2,2),imshow(min(img_white_dm, 1)),title('WhiteDemosaicing');

    % Gray
    r_gray = interp2(img_gw(:,:,1),1);
    g_gray = interp2(img_gw(:,:,2),1);
    b_gray = interp2(img_gw(:,:,3),1);
    img_gray_dm = cat(3, r_gray, g_gray, b_gray);
    subplot(1,2,1),imshow(min(img_gray_dm, 1)),title('GrayDemosaicing');

results
<img width="1161" alt="image" src="https://user-images.githubusercontent.com/75105873/227754795-84a2a98d-38b3-4284-a4bf-9ffacaaaf30f.png">


## BRIGHTNESS ADJUSTMENT AND GAMMA CORRECTION (20 POINTS)
    % White
    coeff = 2;
    img_white_dm_gray = rgb2gray(img_white_dm);
    white_dm_gray_max = max(img_white_dm_gray,[],[1,2]);
    img_white_dm = img_white_dm * 1/white_dm_gray_max * coeff;

    for i=1:3
     for j=1:2855
         for k=1:4289
             if img_white_dm(j,k,i) <= 0.0031308
                 img_white_dm(j,k,i) = img_white_dm(j,k,i) * 12.92;
             else
                 img_white_dm(j,k,i) = (1+0.055) * img_white_dm(j,k,i)^(1/2.4) - 0.055;
             end
         end
     end
    end

    figure;subplot(1,2,1),imshow(min(img_white_dm, 1)),title('WhiteGamma');


    % Gray
    img_gray_dm_gray = rgb2gray(img_gray_dm);
    gray_dm_gray_max = max(img_gray_dm_gray,[],[1,2]);
    img_gray_dm = img_gray_dm * 1/gray_dm_gray_max * coeff;

    for i=1:3
     for j=1:2855
         for k=1:4289
             if img_gray_dm(j,k,i) <= 0.0031308
                 img_gray_dm(j,k,i) = img_gray_dm(j,k,i) * 12.92;
             else
                 img_gray_dm(j,k,i) = (1+0.055) * img_gray_dm(j,k,i)^(1/2.4) - 0.055;
             end
         end
     end
    end

    subplot(1,2,2),imshow(min(img_gray_dm, 1)),title('GrayGamma');

* results
Coefficient : 1*Inverse(max) 
<img width="1162" alt="image" src="https://user-images.githubusercontent.com/75105873/227755262-a6e6908f-df39-4d93-adff-7a60417ff9d7.png">

Coefficient : 2*Inverse(max) 
<img width="1160" alt="image" src="https://user-images.githubusercontent.com/75105873/227755204-be922c71-eae9-484d-ae1a-9eb7a011f0f4.png">


## COMPRESSION (5 PTS) 
    imwrite(img_gray_dm, 'GrayWorld_png.png')
    imwrite(img_white_dm, 'WhiteWorld_png.png')

    for i=10:10:90
        imwrite(img_gray_dm, sprintf('gray_world_%d.jpg', i), 'jpg', 'Quality', i)
        imwrite(img_white_dm, sprintf('white_world_%d.jpg', i), 'jpg', 'Quality', i)
    end
             

* results
If Quality Factor = 90
Gray JPEG : 2,017,602 bytes
Gray PNG : 15,875,711 bytes
Compression ratio : 12.7% 
White JPEG : 2,026,468 bytes
White PNG : 15,833,534 bytes
Compression ratio : 12.7% 

If Qality Factor = 50
Gray JPEG : 721,798 bytes
Gray PNG : 15,875,711 bytes
Compression ratio : 4.55% 
White JPEG : 724,142 bytes
White PNG : 15,833,534 bytes
Compression ratio : 4.57% 
