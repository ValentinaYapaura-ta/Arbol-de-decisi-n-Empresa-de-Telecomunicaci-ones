# Modelo de Árbol de Decisión: Predicción de Aceptación de Oferta Móvil

**Instituto Tecnológico Beltrán (ITB)**

*Unidad 4: Algoritmos de Aprendizaje Supervisado y Métricas de Información*

## Descripción del Problema

Una empresa de telecomunicaciones busca determinar si un cliente aceptará un plan de datos móviles basándose en su perfil y hábitos de consumo. Para ello, se analiza un conjunto de datos histórico de 10 clientes con el objetivo de seleccionar el atributo más discriminante y construir un árbol de decisión optimizado.

## Estructura del Dataset

El conjunto de datos comprende los siguientes atributos sociodemográficos e institucionales:

* **ID:** Identificador único del cliente.
* **Edad:** Valor numérico en años.
* **Uso de Datos:** Volumen de consumo mensual medido en Gigabytes (GB).
* **Tiene Línea Fija:** Atributo categórico booleano (`Sí` / `No`).
* **Aceptó Oferta:** Variable objetivo o clase destino (`Sí` / `No`).

| ID | Edad | Uso de Datos (GB) | Tiene Línea Fija | Aceptó Oferta |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 24 | 2.5 | No | No |
| 2 | 38 | 6.0 | Sí | Sí |
| 3 | 29 | 3.0 | No | No |
| 4 | 45 | 8.0 | Sí | Sí |
| 5 | 52 | 7.5 | Sí | Sí |
| 6 | 33 | 4.0 | No | No |
| 7 | 41 | 5.5 | Sí | Sí |
| 8 | 27 | 2.0 | No | No |
| 9 | 36 | 6.5 | Sí | Sí |
| 10 | 31 | 3.5 | No | No |

## Preprocesamiento y Discretización de Datos

Para evaluar las particiones del árbol, las variables continuas se agrupan en las siguientes categorías parametrizadas:

1. **Rango Etario (Edad):**
   * `Joven`: $\le 30$ años (Clientes 1, 3, 8).
   * `Adulto`: $31$ a $50$ años (Clientes 2, 4, 6, 7, 9, 10).
   * `Mayor`: $> 50$ años (Cliente 5).

2. **Consumo de Datos (Uso de Datos):**
   * `Bajo`: $\le 3.0$ GB (Clientes 1, 3, 8).
   * `Medio`: $3.1$ a $6.0$ GB (Clientes 2, 6, 7, 10).
   * `Alto`: $> 6.0$ GB (Clientes 4, 5, 9).

## Desarrollo Paso a Paso

### 1. Entropía del Conjunto Original $H(S)$

El conjunto total $S$ contiene $10$ registros:
* Clientes que aceptaron la oferta ($p$): $5$
* Clientes que no aceptaron la oferta ($n$): $5$

Fórmula de Entropía de Shannon:
$$H(S) = -p_{\text{Sí}} \cdot \log_2(p_{\text{Sí}}) - p_{\text{No}} \cdot \log_2(p_{\text{No}})$$

$$H(S) = -\left(\frac{5}{10}\right) \cdot \log_2\left(\frac{5}{10}\right) - \left(\frac{5}{10}\right) \cdot \log_2\left(\frac{5}{10}\right)$$
$$H(S) = -0.5 \cdot (-1) - 0.5 \cdot (-1) = 0.5 + 0.5 = \mathbf{1.000}$$

La entropía inicial de $1.000$ representa la máxima incertidumbre del sistema.

### 2. Evaluación de Ganancia de Información por Atributo

La Ganancia de Información $IG(S, A)$ se define como:
$$IG(S, A) = H(S) - \sum_{v \in \text{Valores}(A)} \frac{|S_v|}{|S|} \cdot H(S_v)$$

#### Atributo: Tiene Línea Fija

* **Línea Fija = "Sí" ($5$ casos):** Clientes $2, 4, 5, 7, 9$. Todos aceptaron ($5$ "Sí", $0$ "No").
  $$H(S_{\text{Sí}}) = 0.000$$
* **Línea Fija = "No" ($5$ casos):** Clientes $1, 3, 6, 8, 10$. Ninguno aceptó ($0$ "Sí", $5$ "No").
  $$H(S_{\text{No}}) = 0.000$$

$$H(S | \text{Línea Fija}) = \left(\frac{5}{10}\right) \cdot 0.000 + \left(\frac{5}{10}\right) \cdot 0.000 = 0.000$$
$$IG(S, \text{Línea Fija}) = 1.000 - 0.000 = \mathbf{1.000}$$

#### Atributo: Uso de Datos

* **Bajo ($\le 3$ GB) — $3$ casos:** Clientes $1, 3, 8$ ($0$ "Sí", $3$ "No"). $H(S_{\text{Bajo}}) = 0.000$.
* **Medio ($3.1 - 6$ GB) — $4$ casos:** Clientes $2, 6, 7, 10$ ($2$ "Sí", $2$ "No"). $H(S_{\text{Medio}}) = 1.000$.
* **Alto ($> 6$ GB) — $3$ casos:** Clientes $4, 5, 9$ ($3$ "Sí", $0$ "No"). $H(S_{\text{Alto}}) = 0.000$.

$$H(S | \text{Uso}) = \left(\frac{3}{10}\right) \cdot 0.000 + \left(\frac{4}{10}\right) \cdot 1.000 + \left(\frac{3}{10}\right) \cdot 0.000 = 0.400$$
$$IG(S, \text{Uso}) = 1.000 - 0.400 = \mathbf{0.600}$$

#### Atributo: Edad

* **Joven ($\le 30$) — $3$ casos:** Clientes $1, 3, 8$ ($0$ "Sí", $3$ "No"). $H(S_{\text{Joven}}) = 0.000$.
* **Adulto ($31 - 50$) — $6$ casos:** Clientes $2, 4, 6, 7, 9, 10$ ($4$ "Sí", $2$ "No").
  $$H(S_{\text{Adulto}}) = -\left(\frac{4}{6}\right) \cdot \log_2\left(\frac{4}{6}\right) - \left(\frac{2}{6}\right) \cdot \log_2\left(\frac{2}{6}\right) = 0.918$$
* **Mayor ($> 50$) — $1$ caso:** Cliente $5$ ($1$ "Sí", $0$ "No"). $H(S_{\text{Mayor}}) = 0.000$.

$$H(S | \text{Edad}) = \left(\frac{3}{10}\right) \cdot 0.000 + \left(\frac{6}{10}\right) \cdot 0.918 + \left(\frac{1}{10}\right) \cdot 0.000 = 0.551$$
$$IG(S, \text{Edad}) = 1.000 - 0.551 = \mathbf{0.449}$$

### 3. Comparativa de Ganancia de Información

| Atributo | Entropía Ponderada $H(S|A)$ | Ganancia de Información $IG(S, A)$ | Decisión |
| :--- | :---: | :---: | :---: |
| **Tiene Línea Fija** | **0.000** | **1.000** | **Nodo Raíz Seleccionado** |
| **Uso de Datos** | 0.400 | 0.600 | Descartado |
| **Edad** | 0.551 | 0.449 | Descartado |

## Estructura del Árbol de Decisión Resultante

Dado que la condición `Tiene línea fija` reduce la entropía directamente a $0$, el árbol de decisión consta de un único nivel de prueba sin necesidad de ramas auxiliares:

* **¿Tiene Línea Fija?**
  * **Sí:** Aceptó Oferta ($5$ clientes).
  * **No:** No Aceptó Oferta ($5$ clientes).

## Conclusiones

1. **Mejor Atributo Razonado:** El atributo **`Tiene línea fija`** es la variable óptima para encabezar el nodo raíz del árbol, ya que alcanza el valor máximo posible de Ganancia de Información ($1.000$).
2. **Determinación Predictiva:** Esta variable segmenta el conjunto de datos de forma pura: todo cliente que posee línea fija en la muestra acepta la oferta de plan de datos móviles, mientras que aquellos sin línea fija la rechazan. No se requiere evaluar variables secundarias como la edad o el consumo de datos para este dataset de entrenamiento.