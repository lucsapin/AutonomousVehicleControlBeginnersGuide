# AutonomousVehicleControlBeginnersGuide
[![Linux_CI](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/actions/workflows/Linux_CI.yml/badge.svg)](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/actions/workflows/Linux_CI.yml) [![Windows_CI](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/actions/workflows/Windows_CI.yml/badge.svg)](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/actions/workflows/Windows_CI.yml) [![MacOS_CI](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/actions/workflows/MacOS_CI.yml/badge.svg)](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/actions/workflows/MacOS_CI.yml) [![CodeFactor](https://www.codefactor.io/repository/github/shisatoyano/autonomousvehiclecontrolbeginnersguide/badge)](https://www.codefactor.io/repository/github/shisatoyano/autonomousvehiclecontrolbeginnersguide)  

## Contents
* [Preface](#preface)
* [動作条件](#動作条件)
* [使い方](#使い方)
* [プログラムの構成](#プログラムの構成)
    * [common](#common)
    * [motion_model](#motion_model)
    * [vehicle_drawing](#vehicle_drawing)
* [本プロジェクトへの貢献](#本プロジェクトへの貢献)
* [ライセンス](#ライセンス)
* [お問い合わせ](#お問い合わせ)

## Preface
This repository is beginner's guide to learn basic way of thinking and representative algorithms for Autonomous vehicle control. Explanation documents and sample codes about Motion model, Localization, Path Planning/Following and Vehicle Control are included in this repository. All of them are themed typical 4 wheel drive vehicle. I hope you can understand the above algorithms practically by reading documents and implementing codes.  

## 動作条件
各サンプルプログラムを動かすためには下記条件を満たしている必要があります。  
* Python 3.10.x以上
* numpy
* matplotlib
* pytest
* pytest-cov

## 使い方
1. このリポジトリを自分のPCのローカルにクローンします。  
```bash
$ git clone https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms.git
```
2. 必要なPython用ライブラリをインストールします。  
```bash
$ pip install -r requirements.txt
```
3. 下記のように各ディレクトリにあるPythonスクリプトを実行します。  
```bash
$ python motion_model/linear_motion/linear_motion_model.py
```
4. もし気に入っていただけたら、スターを付けていただけると嬉しいです。

## プログラムの構成
本リポジトリのサンプルプログラムは、以下のような構成でジャンルや目的別のディレクトリに分けられて配置されています。  
### common
プログラム全体で共通で使えるモジュールをまとめています。  
詳細については、コードに書かれたコメントや[こちらのドキュメント](/common/common.md)を参照ください。  

### motion_model
車両の動きを理論的に計算するための運動モデルを実装したモジュールをまとめています。  
各種モデルを実装したクラスのコードは全て単体で実行できるようになっており、実行すると以下のように運動モデルに従った車両の動きがアニメーションで描画されます。  
![](/gif/linear_motion_model.gif)  
また、このディレクトリにある運動モデルクラスをimportすることで、位置計測や車両制御などのアルゴリズムのシミュレーションにも使うことができます。運動モデルの理論の詳細については[こちらのドキュメント](/motion_model/motion_model.md)を参照ください。  

### vehicle_drawing
各種シミュレーションを実行したときの車両の動きをアニメーションとして可視化するモジュールをまとめています。主に二輪モデルと四輪モデルの車両の二種類を描画でき、それぞれはボディやタイヤなどのパーツモジュールの集合体として表現されています。例えば、モジュールの一つであるfour_wheels_vehicle.pyを実行すると、このように四輪モデルの車両の描画例を見ることができます。  
![](/gif/four_wheels_vehicle.gif)  
詳細については、[こちらのドキュメント](/vehicle_drawing/vehicle_drawing.md)を参照ください。  

## 本プロジェクトへの貢献
バグの報告や改善要望などありましたらご自由に[Issues](https://github.com/ShisatoYano/AutonomousDrivingSamplePrograms/issues)に登録ください。  
コード追加や変更などのプルリクエストも大歓迎です。プルリクエストを出していただく際は、[こちらの手順](/docs/contributing_guide.md)に従ってください。また、コード変更時は[こちらのルール](/docs/test_guide.md)に従ってテストを実行し、既存コードに不具合が起きていないことを確認してからプルリクエストを出すようにしてください。  


## お問い合わせ
ご質問等ありましたら、こちらのメールアドレスまでご連絡ください。  
shisatoyano@gmail.com  
