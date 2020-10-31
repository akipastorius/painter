from painter.painter import Painter

painter = Painter(input_path='data/input/aki.png')
painter.scale(25)
painter.paint(stroke_scale=2)
painter.save_canvas()
