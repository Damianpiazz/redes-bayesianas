# redes-bayesianas

Implementación *from scratch* de una **Red Bayesiana (Bayesian Network)**, desarrollada para el Trabajo Práctico de Tecnologías Inteligentes para Explotación de Información. El proyecto permite modelar incertidumbre mediante grafos probabilísticos dirigidos e implementar algoritmos básicos de inferencia sobre variables aleatorias.

---

# Objetivos del Proyecto

Este repositorio tiene como objetivo:

- Implementar una Red Bayesiana desde cero.
- Representar conocimiento mediante grafos acíclicos dirigidos (DAG).
- Modelar relaciones causales entre variables aleatorias.
- Implementar Tablas de Probabilidad Condicional (CPT).
- Calcular probabilidades conjuntas y condicionales.
- Realizar inferencia probabilística utilizando evidencia.
- Comprender el aprendizaje estructural y paramétrico de una red bayesiana.

---

# ¿Qué es una Red Bayesiana?

Una **Red Bayesiana** es un modelo gráfico probabilístico representado mediante un **Grafo Acíclico Dirigido (Directed Acyclic Graph - DAG)**.

En este modelo:

- Cada nodo representa una variable aleatoria.
- Cada arco representa una dependencia o influencia causal.
- Cada nodo almacena una Tabla de Probabilidad Condicional (Conditional Probability Table - CPT).

Las Redes Bayesianas permiten representar incertidumbre y realizar inferencias probabilísticas a partir de evidencia observada.

---

# Fundamentos Teóricos

## Nodo

Cada nodo representa una variable aleatoria que puede ser discreta o continua.

La probabilidad de un estado particular se expresa como:

$$
P(x)=P(X=x)
$$

---

## Arco

Un arco representa una relación de dependencia entre dos variables.

Si existe un arco

$$
(X,Y)
$$

entonces:

- \(X\) es padre de \(Y\).
- \(Y\) depende probabilísticamente de \(X\).

---

## Padre e Hijo

Si existe un arco dirigido

$$
X \rightarrow Y
$$

entonces:

- **X** es el padre de **Y**.
- **Y** es hijo de **X**.

---

## Estados de una Variable

Los estados de una variable deben cumplir dos propiedades:

- Ser mutuamente excluyentes.
- Formar un conjunto exhaustivo.

---

## Probabilidad Conjunta

La probabilidad conjunta representa la probabilidad de ocurrencia simultánea de varias variables.

$$
P(X,Y,\ldots,Z)
$$

Cumple:

$$
\sum P(x_i,y_j,\ldots,z_k)=1
$$

---

## Probabilidad Condicional

La probabilidad de un evento condicionada a otro se define como:

$$
P(Y|X)
=
\frac{P(X,Y)}
{P(X)}
\qquad P(X)>0
$$

---

## Independencia

Dos variables son independientes cuando el conocimiento de una no modifica la probabilidad de la otra.

$$
P(X,Y)=P(X)\cdot P(Y)
$$

Equivalentemente,

$$
P(Y|X)=P(Y)
$$

y

$$
P(X|Y)=P(X)
$$

---

## Probabilidad a Priori

Es la probabilidad de una variable antes de observar evidencia.

$$
P(X)
$$

---

## Probabilidad a Posteriori

Es la probabilidad de una variable luego de incorporar evidencia.

$$
P(X|e)
$$

donde:

- \(e\) representa el conjunto de evidencias observadas.

---

# Teorema de Bayes

Las Redes Bayesianas se fundamentan en el Teorema de Bayes.

$$
P(A|B)=
\frac{
P(B|A)\cdot P(A)
}
{
P(B)
}
$$

Este teorema permite actualizar probabilidades cuando se incorpora nueva evidencia.

---

# Regla del Producto

Una de las propiedades más importantes de una Red Bayesiana es que la distribución conjunta puede factorizarse como:

$$
P(X_1,X_2,\ldots,X_n)
=
\prod_{i=1}^{n}
P(X_i|Parents(X_i))
$$

Esta propiedad reduce considerablemente la cantidad de probabilidades necesarias para representar un sistema complejo.

---

# Representación del Conocimiento

Una Red Bayesiana está compuesta por:

1. Un conjunto de nodos que representan variables aleatorias.
2. Un conjunto de arcos que representan relaciones causales.
3. Una Tabla de Probabilidad Condicional (CPT) asociada a cada nodo.

Si un nodo no posee padres, únicamente almacena probabilidades **a priori**.

---

# Tablas de Probabilidad Condicional (CPT)

Ejemplo:

## Probabilidad de Lluvia

| Lluvia | Probabilidad |
|---------|-------------:|
| Sí | 0.20 |
| No | 0.80 |

---

## Probabilidad del Aspersor

| Lluvia | Aspersor = Sí | Aspersor = No |
|---------|--------------:|--------------:|
| Sí | 0.01 | 0.99 |
| No | 0.40 | 0.60 |

---

# Inferencia Probabilística

Una vez construida la red es posible responder consultas utilizando evidencia.

Ejemplo:

```text
Evidencia

Césped Mojado = Sí

Consulta

P(Lluvia = Sí | Césped Mojado = Sí)
```

Durante la inferencia las probabilidades a priori se actualizan para obtener probabilidades a posteriori.

---

# Aprendizaje en Redes Bayesianas

El aprendizaje consiste en construir automáticamente la red utilizando datos almacenados en una base de datos.

Existen dos enfoques principales:

## Aprendizaje Estructural

Consiste en descubrir automáticamente la estructura del grafo, es decir, determinar qué variables están relacionadas mediante arcos.

---

## Aprendizaje Paramétrico

Consiste en estimar las probabilidades de cada Tabla de Probabilidad Condicional suponiendo que la estructura de la red ya es conocida.

---

# Algoritmo General

## Construcción

1. Definir las variables del dominio.
2. Construir el grafo dirigido.
3. Determinar relaciones padre-hijo.
4. Construir las Tablas de Probabilidad Condicional.

---

## Inferencia

1. Recibir evidencia.
2. Propagar la evidencia por la red.
3. Actualizar probabilidades.
4. Responder consultas probabilísticas.

---

# Pseudocódigo

```text
Crear nodos

Construir el grafo dirigido

Definir relaciones padre-hijo

Crear las tablas CPT

Ingresar evidencia

Actualizar probabilidades

Responder consultas
```

---

# Dataset Utilizado

Como ejemplo se utiliza una red sencilla de diagnóstico.

Archivo:

```text
data/weather.csv
```

## Variables

| Variable | Tipo |
|----------|------|
| Lluvia | Booleana |
| Aspersor | Booleana |
| Césped Mojado | Booleana |

---

# Ejemplo de Inferencia

Consulta:

$$
P(Lluvia|Césped=Mojado)
$$

Resultado esperado:

```text
P(Lluvia = Sí | Césped Mojado = Sí)

0.357
```

---

# Visualizaciones

El proyecto genera gráficos automáticamente al ejecutar `python main.py`. Los archivos se guardan en la carpeta `plots/`.

## Grafo de la Red Bayesiana

```text
plots/weather_network_graph.png
```

Representación visual de la estructura de la red como grafo dirigido acíclico (DAG). Cada nodo es una variable aleatoria y cada arco indica una relación de dependencia causal. Los nodos raíz (sin padres) se muestran en color turquoise, los nodos hijos en rojo. Permite ver rápidamente qué variables influyen sobre otras.

---

## CPT Individuales (Heatmaps)

```text
plots/weather_cpt_Lluvia.png
plots/weather_cpt_Aspersor.png
plots/weather_cpt_Cesped_Mojado.png
```

Heatmaps que muestran las Tablas de Probabilidad Condicional (CPT) de cada variable. Para nodos raíz como **Lluvia**, se muestra un bar chart con las probabilidades a priori. Para nodos con padres como **Aspersor** o **Césped Mojado**, se muestra un heatmap donde las filas representan las combinaciones de valores de los padres y las columnas los estados del hijo. El color indica la magnitud de la probabilidad: tonos más intensos significan mayor probabilidad.

---

## Todas las CPT (Resumen)

```text
plots/weather_cpt_all.png
```

Panel con todas las tablas de probabilidad en una sola imagen. Muestra las probabilidades prior de las raíces como bar charts y las condicionales de los hijos como heatmaps. Útil para tener una visión completa de la red en un solo gráfico.

---

## Comparación Prior vs Posterior

```text
plots/weather_comparison_Lluvia.png
```

Gráfico de barras comparativo que muestra la distribución de probabilidad de una variable **antes** (prior, en turquoise) y **después** (posterior, en rojo) de incorporar evidencia. Permite visualizar cómo cambia el conocimiento al observar datos. Por ejemplo, al observar Césped Mojado = Sí, la probabilidad de Lluvia se actualiza reflejando la nueva información.

---

# Ejemplo de Grafo

```text
        Lluvia
        /    \
       /      \
Aspersor   Humedad
       \      /
        \    /
     Césped Mojado
```

---

# Estructura del Proyecto

```text
redes-bayesianas/
│
├── data/
│   └── weather.csv
│
├── src/
│   ├── node.py
│   ├── bayesian_network.py
│   ├── inference.py
│   ├── probability.py
│   ├── utils.py
│   └── visualization.py
│
├── tests/
│   ├── test_probability.py
│   ├── test_inference.py
│   └── test_network.py
│
├── plots/
│   ├── weather_*.png
│
├── main.py
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Complejidad

Sea:

- \(n\): cantidad de variables.
- \(m\): cantidad de estados posibles.

La inferencia exacta posee una complejidad exponencial en el peor caso.

$$
O(2^n)
$$

Dependiendo de la estructura de la red pueden utilizarse algoritmos más eficientes como **Variable Elimination**, **Belief Propagation** o el algoritmo de **Chow-Liu** para árboles bayesianos.

---

# Aplicaciones

- Diagnóstico médico.
- Sistemas expertos.
- Predicción meteorológica.
- Detección de fraude.
- Bioinformática.
- Robótica.
- Inteligencia Artificial.
- Soporte para la toma de decisiones.
- Análisis de riesgos.

---

# Ventajas

- Permiten representar incertidumbre de forma natural.
- Modelan relaciones causales entre variables.
- Facilitan el razonamiento predictivo y diagnóstico.
- Reducen la complejidad de representar distribuciones conjuntas.
- Son fácilmente interpretables mediante grafos.

---

# Testing

Ejecutar:

```bash
pytest
```

---

# Descubrimientos tras aplicar la Red Bayesiana

Luego de ejecutar la inferencia probabilística sobre el dataset `weather.csv`, se obtienen los siguientes hallazgos:

## 1. Distribución a Priori: Lluvia

La probabilidad a priori de lluvia es:

| Estado | Probabilidad |
|--------|-------------:|
| No | 0.800 |
| Sí | 0.200 |

Esto indica que en el dataset, la lluvia ocurre el **20%** de las veces. Es decir, la mayoría de los días no llueve.

---

## 2. Relación Lluvia → Aspersor

La CPT del Aspersor revela una **correlación inversa fuerte** con la lluvia:

| Lluvia | P(Aspersor=Sí) | P(Aspersor=No) |
|--------|---------------:|---------------:|
| No | 0.575 | 0.425 |
| Sí | 0.000 | 1.000 |

**Descubrimiento**: Cuando llueve, el aspersor **nunca** se activa (P=0). Cuando no llueve, el aspersor se activa con probabilidad 0.575. Esto tiene sentido lógico: si ya está lloviendo, no hay necesidad de regar el césped.

---

## 3. Relación Lluvia + Aspersor → Césped Mojado

La CPT del Césped Mojado considera **dos padres** simultáneamente:

| Lluvia | Aspersor | P(Césped=Sí) | P(Césped=No) |
|--------|----------|--------------:|--------------:|
| No | No | 0.294 | 0.706 |
| No | Sí | 0.652 | 0.348 |
| Sí | No | 0.700 | 0.300 |
| Sí | Sí | 0.000 | 0.000 |

**Descubrimientos**:

- Cuando **no llueve y el aspersor está activado**, la probabilidad de césped mojado sube a **0.652** (vs. 0.294 sin aspersor).
- Cuando **llueve sin aspersor**, la probabilidad de césped mojado es **0.700** (la lluvia es el factor dominante).
- La combinación Lluvia=Sí + Aspersor=Sí **no aparece en el dataset**, por eso ambas probabilidades son 0.

---

## 4. Inferencia: P(Lluvia | Césped Mojado = Sí)

La consulta principal del modelo:

$$
P(Lluvia \mid Césped\_Mojado = Sí)
$$

Resultado:

| Estado | Probabilidad |
|--------|-------------:|
| No | 0.741 |
| Sí | 0.259 |

**Interpretación**: Al observar que el césped está mojado, la probabilidad de que haya llovido **aumenta** de 0.200 (prior) a 0.259 (posterior). Es decir, el césped mojado es una evidencia **débil pero positiva** de lluvia.

---

## 5. Análisis del Impacto de la Evidencia

| Métrica | Prior (sin evidencia) | Posterior (Césped=Sí) | Cambio |
|---------|----------------------:|----------------------:|-------:|
| P(Lluvia=Sí) | 0.200 | 0.259 | +0.059 |
| P(Lluvia=No) | 0.800 | 0.741 | -0.059 |

La evidencia del césped mojado **actualiza** la creencia sobre la lluvia, pero el efecto es moderado. Esto se debe a que el césped puede mojarse por **dos causas independientes**: la lluvia o el aspersor. El aspersor tiene una alta probabilidad de activarse cuando no llueve (0.575), lo que "diluye" la señal de la lluvia.

---

## 6. Conclusión General

La red bayesiana demuestra cómo la **propagación de evidencia** funciona en la práctica:

1. La lluvia es un evento relativamente raro (20%).
2. El aspersor compensa la ausencia de lluvia activándose con frecuencia.
3. El césped mojado puede deberse a múltiples causas, por lo que su observación **no es concluyente** para inferir lluvia.
4. La inferencia bayesiana cuantifica correctamente esta ambigüedad: P(Lluvia=Sí | Césped=Sí) = 0.259, un valor **moderadamente superior** al prior.

Este ejemplo ilustra el poder de las redes bayesianas para razonar bajo incertidumbre cuando existen **múltiples causas** para un mismo efecto.

---

# Referencias

- Judea Pearl (1988). *Probabilistic Reasoning in Intelligent Systems.*
- Daphne Koller & Nir Friedman. *Probabilistic Graphical Models.*
- Russell, S.; Norvig, P. *Artificial Intelligence: A Modern Approach.*
- Chow, C.; Liu, C. (1968). *Approximating Discrete Probability Distributions with Dependence Trees.*