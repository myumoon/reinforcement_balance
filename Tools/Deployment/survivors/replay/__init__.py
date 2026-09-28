"""recorded capture session を実 controller へ決定的に再生する replay パッケージ。

仮想時計(virtual_clock)と記録済み frame source(recorded_frame_source)を controller の
既存注入点へ渡し、実時計・実入力デバイスに触れずに同じ入力から同じ結果を再現します。
"""
