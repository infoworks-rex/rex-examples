# Learning to Drive Smoothly in Minutes

Learning to drive smoothly in minutes, using a reinforcement learning algorithm <span style="color:red">Soft Actor-Critic (SAC)</span> and <span style="color:red">Variational AutoEncoder (VAE)</span> in the Donkey Car simulator.


Blog post on Medium: [link](https://medium.com/@araffin/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)

Video: [https://www.youtube.com/watch?v=iiuKh0yDyKE](https://www.youtube.com/watch?v=iiuKh0yDyKE)


Level-0          | Level-1
:-------------------------:|:-------------------------:
![result](content/smooth.gif)  | ![result](content/level1.gif)
[Download VAE](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) |  [Download VAE](https://drive.google.com/open?id=1hfQNAvVp2QmbmTLklWt2MxtAjrlisr2B)
[Download pretrained agent](https://drive.google.com/open?id=10Hgd5BKfn1AmmVdLlNcDll6yXqVkujoq) | [Download pretrained agent](https://drive.google.com/open?id=104tlsIrtOTVxJ1ZLoTpBDzK4-DRTA5et)

Note: pretrained agents는 `logs/sac/` 폴더에 넣어주어야 합니다. 없으면 생성! (you need to pass `--exp-id 6` (index of the folder) to use the pretrained agent). 그리고 VAE Level 0, VAE Level 1은 `logs` 폴더에 넣어줍니다.

저는 기본 python이 2.7이 깔려있어 python3로 pip3를 사용하여 진행하였습니다.
## Quick Start
저는 리눅스 환경에서 진행하였습니다.

0. Download simulator [here](https://drive.google.com/open?id=1h2VfpGHlZetL5RAPZ79bhDRkvlfuB4Wb) or build it from [source](https://github.com/tawnkramer/sdsandbox/tree/donkey)
1. Install dependencies (pip3 install -r requirements.txt)
2. (optional but recommended) Download pre-trained VAE: [VAE Level 0](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) [VAE Level 1](https://drive.google.com/open?id=1hfQNAvVp2QmbmTLklWt2MxtAjrlisr2B)
3. Train a control policy for 5000 steps using Soft Actor-Critic (SAC)

```
python3 train.py --algo sac -vae path-to-vae.pkl -n 5000
```
위의 path-to-vae.pkl 의 경우 VAE 0 미리 다운받은 거로 하려면 logs/vae-level-0-dim-32.pkl로 설정해주면 됩니다.

4. Enjoy trained agent for 2000 steps

```
python3 enjoy.py --algo sac -vae path-to-vae.pkl --exp-id 0 -n 2000
```
위의 path-to-vae.pkl 의 경우 VAE 0 미리 다운받은 거로 하려면 logs/vae-level-0-dim-32.pkl로 설정해주면 됩니다.

To train on a different level, you need to change `LEVEL = 0` to `LEVEL = 1` in `config.py`

## Train the Variational AutoEncoder (VAE)
여기서부터는 처음부터 직접 하는 방법입니다.
0. Collect images using the teleoperation mode:
첫번째 단계로 원격 조종모드 방향키로 직접 운전하며 도로 image를 충분히 모다.(space bar로 녹화모드 변경 가능) 맵을 한번 이상 완주하며 충분히 모아주세요.
```
python3 -m teleop.teleop_client --record-folder path-to-record/folder/
```

1. Train a VAE:
위에서 모은 image를 이용해서 VAE 모델에 넣어서 학습시킵니다. 2시간 정도 걸립니다.
```
python3 -m vae.train --n-epochs 50 --verbose 0 --z-size 64 -f path-to-record/folder/
```

## Train in Teleoparation Mode
위의 코드까지 마치면 logs폴더에 vae.pkl이 생성됩니다.
```
python3 train.py --algo sac -vae logs/vae.pkl -n 5000 --teleop
```

## Test in Teleoparation Mode 원격 조종
여기서 --exp-id는 여러개의 폴더 중에 0번째 폴더안의 vae버전을 사용하겠다는 뜻입니다. 나중에 여러번 학습을 해 vae.pkl이 6번째 폴더까지 있고 6번째 폴더안의 vae.pkl을 사용하고 싶다면 --exp-id 6이라고 써주면 됩니다.

근데 저는 VAE를 학습시켜 vae.pkl을 logs폴더안에 얻었고, logs/sac/폴더 안에 DonkeyVae-v0-level-0_7 로 7번째 폴더를 얻었기 때문에
```
python3 -m teleop.teleop_client --algo sac -vae logs/vae.pkl --exp-id 7
```
로 실행시켜줍니다.

만약 다운 받은걸로 실행하고 싶다면, pretrained agent는 logs/sac <span style="color:red">폴더에 그냥 그대로 두고</span> logs폴더에 있는 vae-level-0-dim-32.pkl을 연결해주어야 에러 없이 잘 진행됩니다.
```
python3 -m teleop.teleop_client --algo sac -vae logs/vae-level-0-dim-32.pkl --exp-id 6
```

## Explore Latent Space

vae.enjoy_latent 코드에 
```
parser.add_argument('--exp-id',help='Experiment ID (-1: no exp folder, 0: latest)',default=0,type=int)
```
를 main에 추가해줍니다. 몇번째 폴더로 연결할지를 결정하기 위해서 입니다.

```
python3 -m vae.enjoy_latent -vae logs/vae.pkl --exp-id 7
```
코드 실행 시 latent 이미지가 나옵니다.

만약 다운 받은걸로 실행하고 싶다면, pretrained agent는 logs/sac <span style="color:red">폴더에 그냥 그대로 두고</span> logs폴더에 있는 vae-level-0-dim-32.pkl을 연결해주어야 에러 없이 잘 진행됩니다.
```
python3 -m vae.enjoy_latent -vae logs/vae-level-0-dim-32.pkl --exp-id 6
```

## Reproducing Results

To reproduce the results shown in the video, you have to check different values in `config.py`.

### Level 0

`config.py`:

```python
MAX_STEERING_DIFF = 0.15 # 0.1로 하면 더 부드러운 조절 가능하지만 더 많은 시간 소요
MAX_THROTTLE = 0.6 # MAX_THROTTLE = 0.5도 괜찮지만, 0.6으로 더 빠르게 주행
MAX_CTE_ERROR = 2.0 # normal mode에서만 2.0, teleoperation mode에서는 10.0
LEVEL = 0
```

Train in normal mode (smooth control), it takes ~5-10 minutes:
```
python train.py --algo sac -n 8000 -vae logs/vae-level-0-dim-32.pkl
```

Train in normal mode (very smooth control with `MAX_STEERING_DIFF = 0.1`), it takes ~20 minutes:
```
python train.py --algo sac -n 20000 -vae logs/vae-level-0-dim-32.pkl
```

Train in teleoperation mode (`MAX_CTE_ERROR = 10.0`), it takes ~5-10 minutes:
```
python train.py --algo sac -n 8000 -vae logs/vae-level-0-dim-32.pkl --teleop
```

### Level 1

Note: level 1에서는 teleoperation mode만 가능

`config.py`:

```python
MAX_STEERING_DIFF = 0.15
MAX_THROTTLE = 0.5 # 여기서는 커브가 많이 급해서 0.6은 좀 과합니다.
LEVEL = 1
```

Train in teleoperation mode, it takes ~10 minutes:
```
python train.py --algo sac -n 15000 -vae logs/vae-level-1-dim-64.pkl --teleop
```

Note: VAE의 size가 level 0과 1에서 많이 다르지만 크게 영향 없습니다.

## Record a Video of the on-board camera

You need a trained model. For instance, for recording 1000 steps with the last trained SAC agent:
```
python -m utils.record_video --algo sac --vae-path logs/level-0/vae-32-2.pkl -n 1000
```
