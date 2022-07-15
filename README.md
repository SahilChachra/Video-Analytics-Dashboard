<h1 align="center">Video Analytics Tool using YoloV5 and Streamlit</h1>

## :innocent: Motivation
As AI engineers, we love data and we love to see graphs and numbers! So why not project the inference data on some platform to understand the inference better? When a model is deployed on the edge for some kind of monitoring, it takes up rigorous amount of frontend and backend developement apart from deep learning efforts — from getting the live data to displaying the correct output. So, I wanted to replicate a small scale video analytics tool and understand what all feature would be useful for such a tool and what could be the limitations?

## :framed_picture: Demo

https://user-images.githubusercontent.com/37156032/160282244-42f6bd8c-bfc8-47af-8973-d3d199140e44.mp4

## :key: Features

<h3>For detailed insights, do check out my <a href="https://sahilchachra.medium.com/video-analytics-dashboard-for-yolov5-and-deepsort-c5994461cb44">Medium Blog</a></h3>

<ol>
    <li>Choose input source - Local, RTSP or Webcam</li>
    <li>Input class threshold</li>
    <li>Set FPS drop warning threshold</li>
    <li>Option to save inference video</li>
    <li>Input class confidence for drift detection</li>
    <li>Option to save poor performing frames</li>
    <li>Display objects in current frame</li>
    <li>Display total detected objects so far</li>
    <li>Display System stats - Ram, CPU and GPU usage</li>
    <li>Display poor performing class</li>
    <li>Display minimum and maximum FPS recorded during inference</li>
</ol> 

## :dizzy: How to use?
<ol>
    <li>Clone this repo</li>
    <li>Install all the dependencies</li>
    <li>Download deepsort <a href="https://drive.google.com/file/d/1TmZRcYQMemPLGjY62LBXwX8HnI_zxy2W/view?usp=sharing">checkpoint</a> file and paste it in deep_sort_pytorch/deep_sort/deep/checkpoint</li>
    <li>Run -> streamlit run app.py</li>
</ol>

## :star: Recent changelog
<ol>
    <li>Updated yolov5s weight file name in detect() in app.py</li>
    <li>Added drive link to download DeepSort checkpoint file (45Mb).</li>
</ol>

## :exploding_head: FAQs
<ol>
    <li><a href="https://github.com/SahilChachra/Video-Analytics-Dashboard/issues/5">How to use custom Yolov5 weight or DeepSort checkpoint file?</a></li>
    <li><a href="https://github.com/SahilChachra/Video-Analytics-Dashboard/issues/3">Unable to use webcam</a></li>
</ol>

## :heart: Extras
Do checkout the Medium article and give this repo a :star:

## Note
The input video should be in same folder where app.py is. If you want to deploy the app in cloud and use it as a webapp then - download the user uploaded video to temporary folder and pass the path and video name to the respective function in app.py . This is Streamlit bug. Check <a href="https://stackoverflow.com/questions/65612750/how-can-i-specify-the-exact-folder-in-streamlit-for-the-uploaded-file-to-be-save">Stackoverflow</a>.
