f00:
- input: vel, gyro
- spectral normalization: True
- shuffle: once
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False


f01:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: True
- shuffle: once
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False

f02:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: onve
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False

f03:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: every epoche
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False

f04:
- input: vel, gyro, rotation (3 coloums)
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False

f05:
- input: vel, gyro, position
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False

f06:
- input: vel, gyro, position, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: False
- hidden layer: 6
- K Folds: False

f07:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: True
- hidden layer: 6
- K Folds: False

f08:
- input: vel, gyro, rotation (first 2 coloumns), position
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: False

f09:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f10:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f11:
- input: vel, gyro, rotation (first 2 coloumns), position, acceleration
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f12:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f13:
- input: vel, gyro, rotation (first 2 coloumns)
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f14:
- input: vel, gyro, rotation (first 2 coloumns), position, acceleration
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f15:
- input: vel, gyro, rotation (first 2 coloumns), acceleration
- spectral normalization: False
- shuffle: once
- weight decay: True
- Dropout: True
- hidden layer: 6
- K Folds: 10 folds

f16:
- input: vel, gyro, rotation (first 2 coloumns), acceleration, position
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: True
- hidden layer: 6
- K Folds: False

f17:
- input: vel, gyro, rotation (first 2 coloumns), acceleration
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: True
- hidden layer: 6
- K Folds: False

f18:
- input: vel, gyro, acceleration, postition
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: True
- hidden layer: 6
- K Folds: False

f19:
- input: vel, gyro, acceleration
- spectral normalization: False
- shuffle: once
- weight decay: False
- Dropout: True
- hidden layer: 6
- K Folds: False
