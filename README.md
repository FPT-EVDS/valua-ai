# valua-ai
## 1. Config
- Install Anaconda (for manage env)
- Install Python 3.7
- Install Cuda 10.1 ( my nvdia compatible :"> don't know other can use or not )  
- Run requirement file : ( *I think should use conda*)

With pip :
```sh
pip install -r requirements.txt
```
With conda:
```sh
conda env create -f flaskserver_py37.yml
```
## 2. Prepare for model
- Download model : https://drive.google.com/drive/u/1/folders/1ZYvPNFrbV4m9L4FMkzNhc0YSVycbel9l
- Create dic work_space
Set up dictionary like:

```sh
work_space/
        ---> model/
            ---> model_ir_se50.pth
        ---> save/
            ---> model_final.pth
```
## 3. Run Project
### 3.1 Flask server
```sh
python app.py 
```
### 3.2 Run for test model:
- Provide the face images your want to detect in the data/face_bank folder, and guarantee it have a structure like following:

```sh
data/facebank/
        ---> id1/
            ---> id1_1.jpg
        ---> id2/
            ---> id2_1.jpg
        ---> id3/
            ---> id3_1.jpg
           ---> id3_2.jpg
```
:"> WAIT NOT YET :))))))))))) i comment this code already 

## 4. Resource 
Link : https://github.com/TreB1eN/InsightFace_Pytorch
