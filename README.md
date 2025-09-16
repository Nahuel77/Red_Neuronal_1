Primeramente aclarar que mas que especificaciones tecnicas, dejaré aqui algunas consideraciones mas cercanas a lo que involucra las nociones basicas de su funcionamiento.
A modo de recordar lo aprendido aqui paso a explicarme a mi mismo y a quien visite este repo.

Se ha dicho a modo de chiste, que la IA es un nido de bucles "IF".
Dentro de las IA se podria decir sin tanto chiste que la red neuronal diseñada en este repositorio es un proceso de estadisticas aplicadas a matrices (o para python arrays bidimensionales*) de un modo secuencial. Obteniendo asi una ultima matriz capa, donde el valor mas alto es el resultado esperado.

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
<br/>
Epoch 0, Loss: 2.3037<br/>
Epoch 10, Loss: 2.3012<br/>
Epoch 20, Loss: 2.3012<br/>
Epoch 30, Loss: 2.3012<br/>
Epoch 40, Loss: 2.3012<br/>
Epoch 50, Loss: 2.3012<br/>
Epoch 60, Loss: 2.3012<br/>
Epoch 70, Loss: 2.3012<br/>
Epoch 80, Loss: 2.3012<br/>
Epoch 90, Loss: 2.3012<br/>
Loss: 2.3011508437188146 Una mierda<br/>
<br/>
ChatGPT me sugerio cambiar a ReLU. Y aunque sabia que la funcion sigmoide estaba obsoleta en este campo, la habia elegido para comenzar a aprender redes neuronales. En resumen, no valio la pena. Pero es bueno comprender lo que es una funcion de gradientes aplicadas a este campo, para saber en que direccion debe moverse la red para obtener el resultado correcto.

Llegado a este punto, combiene aclarar que los resultados de la red, no son una matriz con el valor 1 en el resultado esperado y 0 en el resto.
El resultado es un array de 10 elementos, con un valor entre 0 y 1 que representan la probabilidad de que cada elemento sea el resultado correcto... imaginemos que se le da a la red la entrada de datos correspondiente al numero 3. El resultado esperado de la capa pordia asimilarse a esto:
output_probs = [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]
Y al ser el valor mas alto el del indice 3, 0.85, la red asumirá que el numero es el 3.

Entre la capa de entrada (con 784 neuronas) y la capa oculta siguiente (128 neuronas) existen 100352 conexiones posibles.
La relacion entre estas capas puede calcuarse y pensarse estos resultados, como un nuevo array bidimencional o matriz.
Dicha relacion se establece de la siguiente manera:
Neurona_inicio_x * Wi + Bi, donde W es el peso que tiene dicha relacion y B un sesgo de activacion para la neurona resultante.

En nuestro set de datos de entrenamiento, que viene en formato cvs, tenemos una primer columna "label" que nos indica que numero se representará con las siguientes columnas "pixel_x" (pixel_1, pixel_2, pixel_3...)
De manera que en donde haya un trazo de escritura, se representara con un numero, y donde no, un cero.
Estos datos se preparan separando la columna en un nuevo arreglo y_train.
y_train = train['label'].values
y en una matriz con los valores de los pixeles, sin esta columna. x_train
x_train = train.drop(columns=['label']).values / 255.0
Como el valor maximo es de 255, se normalizan los datos dividiendolos por 255 para obtener como numero maximo un 1 y el resto flotantes mayores a 0. Es decir, x_train contiene numeros entre 0 y 1.

Luego se inicia una matriz que de tamaño 784x128 (100352) inicializada con numeros al azar
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
Esta es la matriz de pesos asociados a las conexiones. Se inicializan de manera aleatoria. Pero con la etapa de entrenamiento esos valores se iran ajustando.
Tambien se inicializa un array de tamaño 128 (128 para esta primer capa) completada con ceros. Este es el array b1 que contiene los sesgos de activacion para cada neurona.
b1 = np.zeros((1, hidden_size))
De igual manera, se ajustan durante el entrenamiento.

Por lo tanto de los valores obtenido de pixel_x * Wi + bi se forma la nueva matriz representante de la primer capa oculta.
Z1=np.dot(x_train, W1)+b1

Aqui, luego, procesaba esta matriz con la funcion sigmoide para comprimir los valores a una escala entre 0 y 1. Pero como dije antes, eso se cambio a la funcion ReLU.

La fucion sigmoide, como dije antes comprimia los valores en una escala entre 0 y 1. En cambio la funcion relu retorna 0 cuando un valor es menor a 0. Y retorna x cuando el valor x es mayor a 0.

La matriz obtenida entonces, con valores superiores a 0, resulta en la capa neuronal A1
    Z1=np.dot(x_train, W1)+b1
    A1=relu(Z1)
El proceso entonces se repite la cantidad de veces correspondiente a la cantidad de capas neuronales.

Finalmente obtengo la capa final, no aplicando la fucion ReLU, que es solo para las capas ocultas. Sino la funcion Softmax.

Softmax calcula la matriz de probabilidades. En esta matriz el numero de valor mas alto es el resultado que se estima correcto.

Softmax: ![alt text](/miscellaneous/image.png)

Softmax tomará la matriz resultante de la capa neuronal oculta previa a la salida, y tomara cada elemento de esa matriz, usandolo como exponente para elevar e a ese exponente y lo dividirá por la sumatoria de todos e elevado a cada elemento de esa misma matriz. (arriba en la imagen la formula matematica).



* No se que variacion tiene en python un array bidimencional, una matriz o un mapa. Respecto al tipo de datos. Si hablo de uno a otro, quiero que se entienda que no hago diferenciación, ya que lo dejo en plano algebraico. No quiere decir que suponga que son el mismo tipo de datos.