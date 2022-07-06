"""
Vehicle chassis drawing program

Author: Shisato Yano
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# 他のディレクトリにあるモジュールを読み込むためのパス設定
sys.path.append("../common")
from transformation import rotate_translate_2d


class Chassis:
    """
    車両のシャーシを描画するクラス
    """

    def __init__(self, axes, front_length_m, rear_length_m):
        """
        コンストラクタ
        axes: 描画オブジェクト
        front_length_m: 車両位置から前方への長さ[m]
        rear_length_m: 車両位置から後方への長さ[m]
        """

        # パラメータのセット
        self.front_length = front_length_m
        self.rear_length = rear_length_m
        
        # シャーシの形を形成するための点群
        self.points = np.array([
            [front_length_m, -rear_length_m],
            [0.0, 0.0]
        ])

        # 描画オブジェクトの初期化
        self.plot, = axes.plot(self.points[0, :], self.points[1, :], lw=1, color='k')
    
    def draw(self, x_m, y_m, yaw_angle_deg):
        """
        シャーシの形を描画する関数
        指定した分だけ回転 + 並進移動させて描画する
        x_m: X軸方向の並進移動量
        y_m: Y軸方向の並進移動量
        yaw_angle_deg: 車両の方位角度[deg]
        """
        
        # 車両の方位角度分だけ回転
        # 車両の移動量分だけ並進移動
        transformed_points = rotate_translate_2d(self.points, x_m, y_m, np.deg2rad(yaw_angle_deg))
        
        # 描画
        self.plot.set_data(transformed_points[0, :], transformed_points[1, :])
