import stepByStep

stepByStep.get_spectrograms('zero.wav', '\'zero\' with 25 filters', num_filters=25)
stepByStep.get_spectrograms('zero.wav', '\'zero\' with 30 filters', num_filters=30)
stepByStep.get_spectrograms('zero.wav', '\'zero\' with 40 filters', num_filters=40)

stepByStep.get_spectrograms('three.wav', '\'three\' instance 1')
stepByStep.get_spectrograms('three2.wav', '\'three\' instance 2')
stepByStep.get_spectrograms('thee_PengWang.wav', '\'three\' instance 3 (Peng Wang)')
