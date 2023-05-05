## 1. Toy Problem
* I implemented 3 objectives stated in A3.pdf and reconstructed the toy image.

      toyim = im2double(imread('data/toy_problem.png')); 
      im_out = toy_reconstruct(toyim);
      disp(['Error: ' num2str(sqrt(sum(toyim(:)-im_out(:))))])

      % Function for reconstruction
      function [ recon ] = toy_reconstruct( img )
          [h, w, nn] = size(img);
          Im2var = zeros(h, w);
          Im2var(1:h*w) = 1: h*w;

          A = sparse([], [], [], 2 * h * w + 1, h * w);
          b = zeros(2 * h * w + 1, 1);

          % (1) Minimize the difference between the x-gradients of v and the x-gradients of s
          e = 1;
          for y = 1:h
              for x = 1:w-1
                  A(e, Im2var(y,x+1)) = 1;
                  A(e, Im2var(y,x)) = -1;
                  b(e) = img(y,x+1)-img(y,x);  
                  e = e+1;
              end  
          end    

          % (2) Minimize the difference between the y-gradients of v and the y-gradients of s
          for y = 1:h-1
              for x = 1:w
                  A(e, Im2var(y+1,x)) = 1;
                  A(e, Im2var(y,x)) = -1;
                  b(e) = img(y+1,x)-img(y,x);  
                  e = e+1;
              end  
          end

          % (3) Minimize the difference between the colors of the top left corners of the two images
          A(e, Im2var(1,1)) = 1;
          b(e) = img(1,1);

          v = A\b;
          recon = reshape(v, [h, w]);
          imwrite(recon, 'results/toy_recon.png');

      end


* Original img

![toy_problem](https://user-images.githubusercontent.com/75105873/236471469-e450b8ff-89e4-4a3f-b8a2-2b4531622097.png)


* Reconstructed img

![toy_recon](https://user-images.githubusercontent.com/75105873/236471433-d75e6c83-a1f9-4188-8496-7012e6ae93fe.png)


## 2, 3. Poisson Blending & Blending with Mixed Gradients
* I implemented objectives stated in A3.pdf for each Poisson and Mixed blending. I applied two methods on penguin chick and penguin images.

      % Applying on chick image
      img_background = imresize(im2double(imread('data/hiking.jpg')), 0.5, 'bilinear');
      img_object = imresize(im2double(imread('data/penguin-chick.jpeg')), 0.5, 'bilinear');
      img_name = 'chick';

      objmask = getMask(img_object);
      [img_s, mask_s] = alignSource(img_object, objmask, img_background);

      im_blend = poissonBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend)

      im_blend = mixedBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend);

      % Applying on penguin image
      img_background = imresize(im2double(imread('data/hiking.jpg')), 0.5, 'bilinear');
      img_object = imresize(im2double(imread('data/penguin.jpg')), 0.5, 'bilinear');
      img_name = 'penguin';

      objmask = getMask(img_object);
      [img_s, mask_s] = alignSource(img_object, objmask, img_background);

      im_blend = poissonBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend)

      im_blend = mixedBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend);

      % Function for Poisson Blending
      function [ img_blend ] = poissonBlend( img_s, mask_s, img_background, img_name )
          [h, w, nn] = size(img_s);
          Im2var = zeros(h, w);

          [yy, xx] = find(mask_s > 0);
          nz = sum(sum(mask_s));

          e = 1;
          for i = 1:nz
              Im2var(yy(i),xx(i)) = e;
              e = e+1;
          end

          A = sparse([], [], []);
          b = zeros(nz, nn);

          e = 1;

          img_blend = img_background; 


          for i = 1:nz
              y = yy(i);
              x = xx(i);
              A(e, Im2var(y,x)) = 4;

              % At pixel (y+1,x)
              % the first part of Equation (1) 
              if mask_s(y+1,x) == 1
                  A(e, Im2var(y+1,x)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y+1,x,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s;
              % the second part of Equation (2)
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y+1,x,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s + reshape(img_background(y+1,x,:), 1, nn);
              end

              % At pixel (y-1,x)
              if mask_s(y-1,x) == 1
                  A(e, Im2var(y-1,x)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y-1,x,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s;
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y-1,x,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s + reshape(img_background(y-1,x,:), 1, nn);
              end

              % At pixel (y,x+1)
              if mask_s(y,x+1) == 1
                  A(e, Im2var(y,x+1)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x+1,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s;
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x+1,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s + reshape(img_background(y,x+1,:), 1, nn);
              end

              % At pixel (y,x-1)
              if mask_s(y,x-1) == 1
                  A(e, Im2var(y,x-1)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x-1,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s;
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x-1,:), 1, nn);
                  b(e,:) = b(e,:) + grad_s + reshape(img_background(y,x-1,:), 1, nn);
              end

              %end
              e = e+1;
          end

          v = A\b;

          e = 1;
          for i=1:nz
              y = yy(i);
              x = xx(i);
              img_blend(y,x,:) = v(e,:);
              e = e + 1;
          end

          imwrite(img_s, strcat('results/', img_name, '_img_s.png'));
          imwrite(mask_s, strcat('results/', img_name, '_mask_s.png'));
          imwrite(img_blend, strcat('results/', img_name, '_poisson_img_blend.png'));

      end


      % Function for Mixed Blending
      function [ img_blend ] = mixedBlend( img_s, mask_s, img_background, img_name )

          [h, w, nn] = size(img_s);
          Im2var = zeros(h, w);

          [yy, xx] = find(mask_s > 0);
          nz = sum(sum(mask_s));

          e = 1;
          for i = 1:nz
              Im2var(yy(i),xx(i)) = e;
              e = e+1;
          end

          A = sparse([], [], []);
          b = zeros(nz, nn);

          e = 1;

          img_blend = img_background; 


          for i = 1:nz
              y = yy(i);
              x = xx(i);
              A(e, Im2var(y,x)) = 4;

              % At pixel (y+1,x)
              % the first part of Equation (1) 
              if mask_s(y+1,x) == 1
                  A(e, Im2var(y+1,x)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y+1,x,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y+1,x,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d;
              % the second part of Equation (1)
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y+1,x,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y+1,x,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d + reshape(img_background(y+1,x,:), 1, nn);
              end

              % At pixel (y-1,x)
              if mask_s(y-1,x) == 1
                  A(e, Im2var(y-1,x)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y-1,x,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y-1,x,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d;
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y-1,x,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y-1,x,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d + reshape(img_background(y-1,x,:), 1, nn);
              end

              % At pixel (y,x+1)
              if mask_s(y,x+1) == 1
                  A(e, Im2var(y,x+1)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x+1,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y,x+1,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d;
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x+1,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y,x+1,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d + reshape(img_background(y,x+1,:), 1, nn);
              end

              % At pixel (y,x-1)
              if mask_s(y,x-1) == 1
                  A(e, Im2var(y,x-1)) = -1;
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x-1,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y,x-1,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d;
              else
                  grad_s = reshape(img_s(y,x,:) - img_s(y,x-1,:), 1, nn);
                  grad_b = reshape(img_background(y,x,:) - img_background(y,x-1,:), 1, nn);
                  if abs(grad_s) > abs(grad_b)
                      grad_d = grad_s;
                  else
                      grad_d = grad_b;
                  end
                  b(e,:) = b(e,:) + grad_d + reshape(img_background(y,x-1,:), 1, nn);
              end

              e = e+1;
          end

          v = A\b;

          e = 1;
          for i=1:nz
              y = yy(i);
              x = xx(i);
              img_blend(y,x,:) = v(e,:);
              e = e + 1;
          end

          imwrite(img_blend, strcat('results/', img_name, '_mixed_img_blend.png'));

      end


![chick_img_s](https://user-images.githubusercontent.com/75105873/236473608-bbf32fcd-a6f2-43ce-bc8b-e8be53f97f70.png)
![chick_mask_s](https://user-images.githubusercontent.com/75105873/236473630-3dbbecf0-6afc-412f-a80b-9b9e0afc425e.png)
* Poisson blending
![chick_poisson_img_blend](https://user-images.githubusercontent.com/75105873/236473658-040cd675-afde-48eb-8121-b5130ed5477d.png)
* Mixed blending
![chick_mixed_img_blend](https://user-images.githubusercontent.com/75105873/236473675-2ecc04ee-025a-4218-95f3-b1f4df8a65ce.png)


![penguin_img_s](https://user-images.githubusercontent.com/75105873/236473817-32e859fa-e613-43c2-88f0-71c3b702471e.png)
![penguin_mask_s](https://user-images.githubusercontent.com/75105873/236473853-9a20552c-0954-40a7-a15f-429345ffd347.png)
* Poisson blending
![penguin_poisson_img_blend](https://user-images.githubusercontent.com/75105873/236473964-b3695ef4-f802-48ba-8651-5d096ae2be4f.png)
* Mixed blending
![penguin_mixed_img_blend](https://user-images.githubusercontent.com/75105873/236473994-04d51e6d-d4e6-4118-bb68-531ac96030a5.png)




## 4. My own examples
* I applied above 2 methods on my own examples.

      % Applying on tiger image
      img_background = imresize(im2double(imread('data/africa.jpeg')), 0.5, 'bilinear');
      img_object = imresize(im2double(imread('data/tiger.jpeg')), 0.5, 'bilinear');
      img_name = 'tiger';

      objmask = getMask(img_object);
      [img_s, mask_s] = alignSource(img_object, objmask, img_background);

      im_blend = poissonBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend)

      im_blend = mixedBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend);


      % Applying on horse image
      img_background = imresize(im2double(imread('data/snow mountain.jpeg')), 0.5, 'bilinear');
      img_object = imresize(im2double(imread('data/horse.jpeg')), 0.5, 'bilinear');
      img_name = 'horse';

      objmask = getMask(img_object);
      [img_s, mask_s] = alignSource(img_object, objmask, img_background);

      im_blend = poissonBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend)

      im_blend = mixedBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend);


      % Applying on retreiver image
      img_background = imresize(im2double(imread('data/park.png')), 0.5, 'bilinear');
      img_object = imresize(im2double(imread('data/retreiver.jpeg')), 0.5, 'bilinear');
      img_name = 'retreiver';

      objmask = getMask(img_object);
      [img_s, mask_s] = alignSource(img_object, objmask, img_background);

      im_blend = poissonBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend)

      im_blend = mixedBlend(img_s, mask_s, img_background, img_name);
      figure(3), hold off, imshow(im_blend);
      

![tiger_img_s](https://user-images.githubusercontent.com/75105873/236474181-f3ce858c-bf79-4b64-bd7d-4bdb6cb61305.png)
![tiger_mask_s](https://user-images.githubusercontent.com/75105873/236474206-718a905a-92fa-48d2-9fb1-e38e90431d55.png)
* Poisson blending
![tiger_poisson_img_blend](https://user-images.githubusercontent.com/75105873/236474221-90500996-06a9-4fd1-a730-a58caa61e0ac.png)
* Mixed blending
![tiger_mixed_img_blend](https://user-images.githubusercontent.com/75105873/236474418-e41c4b9c-3fbb-4bd5-89aa-f4ea1bdf0d48.png)


![horse_img_s](https://user-images.githubusercontent.com/75105873/236474459-e34c18d7-df22-420b-b815-2d0fcf923d07.png)
![horse_mask_s](https://user-images.githubusercontent.com/75105873/236474480-37ed3341-c648-40a1-8190-0a477b1c3b0f.png)
* Poisson blending
![horse_poisson_img_blend](https://user-images.githubusercontent.com/75105873/236474499-8d501db8-18da-41bb-bc5d-27bfd4519c5b.png)
* Mixed blending
![horse_mixed_img_blend](https://user-images.githubusercontent.com/75105873/236474508-2a1a53a1-b4de-49b9-88a9-2f5c98e56ecb.png)


![retreiver_img_s](https://user-images.githubusercontent.com/75105873/236474623-d8f80885-c18a-4871-aab3-cabf3ff0acbf.png)
![retreiver_mask_s](https://user-images.githubusercontent.com/75105873/236474640-799e0f35-b71f-4204-91b7-09e7f56eb207.png)
* Poisson blending
![retreiver_poisson_img_blend](https://user-images.githubusercontent.com/75105873/236474674-87bdbf4b-9d29-43a8-b58d-89b442cd0086.png)
* Mixed blending
![retreiver_mixed_img_blend](https://user-images.githubusercontent.com/75105873/236474734-752f16ef-d817-4a2a-a38b-f49b96449114.png)


Poisson blending has better results compared to mixed blending in that mixed blending outputs are too transparent. It is critical to choose object image and background image which share similar texture.
For the above results, penguin and horse results are the best because both the object and background have white color.
