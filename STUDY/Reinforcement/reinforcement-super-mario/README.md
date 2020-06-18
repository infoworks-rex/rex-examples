run main.py

25000번 학습시 2일 걸렸습니다.

10만번 정도 학습 해야 볼만한 결과가 나올 것 같아 epoch를 10만으로 수정했습니다.

학습 완료 후 play 하는 모습을 보기위해서는 main.py 에서 agent.learn() 을 agent.replay(env=env,model_path=/models, n_replay=100,plot=False) 로 바꾸면 됩니다. plot=True로 하면 출력 과정 중 에러가 발생하여

결과를 보기 위해서는 에러를 해결해야 합니다.

replay 할때 agent.py 코드에서 한번 돌 때 마다 env.reset()을 해주지 않으면 한번 play하고 에러 발생하기 때문에 꼭 추가해 주어야 합니다.



마리오 코드 분석 참고 사이트 링크 : https://sunghan-kim.github.io/ml/3min-dl-ch12/#1231-%ED%95%84%EC%9A%94%ED%95%9C-%EB%AA%A8%EB%93%88-import

