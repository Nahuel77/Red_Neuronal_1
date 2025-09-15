Primeramente aclarar que mas que especificaciones tecnicas, dejaré aqui algunas consideraciones mas cercanas a lo que involucra las nociones basicas de su funcionamiento.
A modo de recordar lo aprendido aqui paso a explicarme a mi mismo y a quien visite este repo.

Se ha dicho a modo de chiste, que la IA es un nido de bucles "IF".
Dentro de las IA se podria decir sin tanto chiste que la red neuronal diseñada en este repositorio es un proceso de estadisticas aplicadas a matrices (o para python arrays bidimensionales) secuencialmente. Obteniendo asi una ultima matriz capa, donde el valor mas alto es el resultado esperado.

Basicamente lo que aqui se codifica, es un grupo de matrices, que se interrelacionan, pero que al momento de entrenarlas, se les da las respuestas para que luego se ajusten esas relaciones.

Esta red neuronal pretende reconocer los numeros del 0 al 9, alimentadas con la informacion de numeros escritos a mano. Por lo tanto la capa de salidas sera un array de 1x10. Es decir los diez resultados esperados.

En cuanto a la capa de inicio, un array de tamaño 784 (28*28), es la que recibe la informacion procesada de una imagen de 28x28 pixeles que contiene el numero escrito a mano.
Basicamente podemos imaginar que dividimos la imagen en 784 celdas y se le da un valor 0 donde no hay tinta, y otro valor donde si la hay.

Lo que ocurre entre esta primera capa de inicio y la capa de salida con el resultado esperado, es un asunto de probabilidades.

Inicie esta red con una capa de inicio de 784 y una sola capa intermedia de 64 neuronas.
(784) -> (64) -> (10)
Y una funcion sigmoide para ajustar los valores obtenidos de las entradas, sus pesos y umbrales, en valores entre 0 y 1 (que sin incluir el 0 y 1), y llevarlos a la matriz que resultaba en la capa siguiente.
Y aunque resulto, la curva de aprendizaje solo fue optima a valores altos de learning_rate. Lo que no me hizo esperar a obtener problemas de estancamiento para comprender que necesitaba una nueva capa neuronal.

Asi fué que agregue re-ajuste la primera capa intermedia a 128 y agregue una nueva capa de 64.
(784) -> (128) -> (64) -> (10)
Que me dio problemas con la función sigmoide. Basicamente la gradiente aplicada a los valores de la capa de salida eran minimos, lo que estancaba la curva de aprendizaje.

Epoch 0, Loss: 2.3037
Epoch 10, Loss: 2.3012
Epoch 20, Loss: 2.3012
Epoch 30, Loss: 2.3012
Epoch 40, Loss: 2.3012
Epoch 50, Loss: 2.3012
Epoch 60, Loss: 2.3012
Epoch 70, Loss: 2.3012
Epoch 80, Loss: 2.3012
Epoch 90, Loss: 2.3012
Loss: 2.3011508437188146 Una mierda

ChatGPT me sugerio cambiar a ReLU. Y aunque sabia que la funcion sigmoide estaba obsoleta en este campo, la habia elegido para comenzar a aprender redes neuronales. En resumen, no valio la pena. Pero es bueno comprender lo que es una funcion de gradientes aplicadas a este campo, para saber en que direccion debe moverse la red para obtener el resultado correcto.

Llegado a este punto, combiene aclarar que los resultados de la red, no son una matriz con el valor 1 en el resultado esperado y 0 en el resto.
El resultado es un array de 10 elementos, con un valor entre 0 y 1 que representan la probabilidad de que cada elemento sea el resultado correcto... imaginemos que se le da a la red la entrada de datos correspondiente al numero 3. El resultado esperado de la capa pordia asimilarse a esto:
output_probs = [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]
Y al ser el valor mas alto el del indice 3, 0.85, la red asumirá que el numero es el 3

