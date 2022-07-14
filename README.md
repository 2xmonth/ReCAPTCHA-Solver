this is my first project involving image classification. i plan on eventually redoing all of this with the knowledge i gained from this.

this solver is pretty bad, it gets some stuff wrong that it really shouldnt, but it does some stuff really well.
generally it takes around a minute to solve a captcha (which is really bad i know)

it uses SCUNet from https://github.com/cszn/SCUNet for image denoising and super res. THIS IS NOT NEEDED AT ALL, i tested this with and without it and it performed pretty much the same. i just kept it in because i think its cool. i would recommend removing it though

i retrained the yolov5x6 model (overkill, if anyone is doing this i would recommend using a smaller model. i just used it because it gave better results and i only had 30 ms inf times) on a dataset of 4000 labeled images from my extractor. (i would have done more but i got tired of labelling images


if you are going to use this i would recommend optimizing the sleeps more as they are probably higher than they need to be.
i would also recommend remaking the fading captcha "system" as it really sucks 


i also started work on a fully requests based solver but i do not plan on releasing it
