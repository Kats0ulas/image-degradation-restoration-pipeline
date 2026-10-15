**Best parameter per filter family (criterion: PSNR)**

| Image | Degradation | Family | Best parameter | PSNR before | PSNR after | ΔPSNR | SSIM after | ΔSSIM | Sharpness (% of reference) |
|---|---|---|---|---|---|---|---|---|---|
| Cameraman (strong edges & texture) | Gauss σ=10 | Gaussian (linear) | Gaussian 3×3 | 28.22 | 30.23 | 2.0044 | 0.8106 | 0.2005 | 20.33 |
| Cameraman (strong edges & texture) | Gauss σ=10 | Median (order statistic) | Median 3×3 | 28.22 | 29.26 | 1.0326 | 0.7514 | 0.1413 | 39.60 |
| Cameraman (strong edges & texture) | Gauss σ=10 | Bilateral (edge-preserving) | Bilateral σc=25, σs=25 | 28.22 | 32.09 | 3.8626 | 0.8654 | 0.2553 | 64.71 |
| Cameraman (strong edges & texture) | Gauss σ=25 | Gaussian (linear) | Gaussian 5×5 | 20.58 | 27.17 | 6.5975 | 0.6468 | 0.3449 | 13.82 |
| Cameraman (strong edges & texture) | Gauss σ=25 | Median (order statistic) | Median 5×5 | 20.58 | 26.31 | 5.7343 | 0.6091 | 0.3071 | 25.26 |
| Cameraman (strong edges & texture) | Gauss σ=25 | Bilateral (edge-preserving) | Bilateral σc=50, σs=50 | 20.58 | 27.86 | 7.2810 | 0.6739 | 0.3720 | 71.07 |
| Cameraman (strong edges & texture) | S&P d=2% | Gaussian (linear) | Gaussian 3×3 | 21.76 | 27.64 | 5.8768 | 0.7006 | 0.0715 | 35.71 |
| Cameraman (strong edges & texture) | S&P d=2% | Median (order statistic) | Median 3×3 | 21.76 | 30.40 | 8.6417 | 0.8669 | 0.2377 | 35.21 |
| Cameraman (strong edges & texture) | S&P d=2% | Bilateral (edge-preserving) | Bilateral σc=100, σs=50 | 21.76 | 26.45 | 4.6855 | 0.6714 | 0.0423 | 33.05 |
| Cameraman (strong edges & texture) | S&P d=10% | Gaussian (linear) | Gaussian 7×7 | 14.75 | 23.92 | 9.1653 | 0.5127 | 0.3210 | 7.9512 |
| Cameraman (strong edges & texture) | S&P d=10% | Median (order statistic) | Median 3×3 | 14.75 | 29.58 | 14.83 | 0.8579 | 0.6662 | 40.23 |
| Cameraman (strong edges & texture) | S&P d=10% | Bilateral (edge-preserving) | Bilateral σc=100, σs=50 | 14.75 | 21.45 | 6.6984 | 0.3662 | 0.1744 | 315.39 |
| Moon (smooth gradients, low contrast) | Gauss σ=10 | Gaussian (linear) | Gaussian 7×7 | 28.12 | 37.53 | 9.4097 | 0.9146 | 0.4704 | 6.1582 |
| Moon (smooth gradients, low contrast) | Gauss σ=10 | Median (order statistic) | Median 5×5 | 28.12 | 36.39 | 8.2715 | 0.8816 | 0.4374 | 33.09 |
| Moon (smooth gradients, low contrast) | Gauss σ=10 | Bilateral (edge-preserving) | Bilateral σc=25, σs=25 | 28.12 | 37.94 | 9.8202 | 0.9066 | 0.4624 | 65.00 |
| Moon (smooth gradients, low contrast) | Gauss σ=25 | Gaussian (linear) | Gaussian 9×9 | 20.17 | 33.82 | 13.64 | 0.8191 | 0.6915 | 9.4358 |
| Moon (smooth gradients, low contrast) | Gauss σ=25 | Median (order statistic) | Median 9×9 | 20.17 | 33.52 | 13.35 | 0.8354 | 0.7078 | 31.11 |
| Moon (smooth gradients, low contrast) | Gauss σ=25 | Bilateral (edge-preserving) | Bilateral σc=100, σs=50 | 20.17 | 34.00 | 13.82 | 0.8229 | 0.6952 | 48.45 |
| Moon (smooth gradients, low contrast) | S&P d=2% | Gaussian (linear) | Gaussian 9×9 | 22.91 | 35.37 | 12.46 | 0.8732 | 0.3938 | 6.1353 |
| Moon (smooth gradients, low contrast) | S&P d=2% | Median (order statistic) | Median 3×3 | 22.91 | 40.42 | 17.51 | 0.9643 | 0.4848 | 33.58 |
| Moon (smooth gradients, low contrast) | S&P d=2% | Bilateral (edge-preserving) | Bilateral σc=100, σs=50 | 22.91 | 36.31 | 13.40 | 0.8950 | 0.4155 | 28.71 |
| Moon (smooth gradients, low contrast) | S&P d=10% | Gaussian (linear) | Gaussian 9×9 | 15.91 | 30.28 | 14.37 | 0.6846 | 0.6151 | 21.16 |
| Moon (smooth gradients, low contrast) | S&P d=10% | Median (order statistic) | Median 3×3 | 15.91 | 39.72 | 23.82 | 0.9646 | 0.8951 | 52.66 |
| Moon (smooth gradients, low contrast) | S&P d=10% | Bilateral (edge-preserving) | Bilateral σc=100, σs=50 | 15.91 | 29.49 | 13.59 | 0.6108 | 0.5413 | 545.69 |
