# Seam Carving for Content-Aware Image Resizing
## Application of Graph Theory in Image Processing

**Author:** Wissem Bahrouni  
**Date:** January 17, 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Energy Map](#energy-map)
4. [Graph Representation](#graph-representation)
5. [Shortest Path Formulation](#shortest-path-formulation)
6. [Dynamic Programming Approach](#dynamic-programming-approach)
7. [Seam Extraction](#seam-extraction)
8. [Seam Removal](#seam-removal)
9. [Complexity Analysis](#complexity-analysis)
10. [Results and Discussion](#results-and-discussion)
11. [Limitations and Improvements](#limitations-and-improvements)
12. [Conclusion](#conclusion)
13. [References](#references)

---

## Introduction

Traditional image resizing techniques such as uniform scaling treat all pixels equally. As a result, important visual elements may become distorted when an image is resized.

Seam Carving is a content-aware image resizing algorithm introduced by Avidan and Shamir (2007). Instead of removing entire rows or columns, the algorithm removes connected paths of pixels with the lowest visual importance.

This method is widely used in computer graphics applications, including Adobe Photoshop's *Content-Aware Scaling* feature.

The objective of this project is to implement the Seam Carving algorithm and demonstrate how graph theory and shortest-path algorithms can be applied to image processing.

---

## Problem Definition

Given an image of height *H* and width *W*, the goal is to reduce its width while preserving the most important visual structures.

A **vertical seam** is defined as a connected path of pixels:

```
S = {(x_y, y) | y = 0, 1, ..., H-1}
```

subject to:

```
|x_y - x_{y-1}| ≤ 1
```

Each seam contains exactly one pixel per image row.

The optimal seam is the one minimizing the total energy:

```
E(S) = Σ E(x_y, y) for y = 0 to H-1
```

---

## Energy Map

### Energy Concept

The importance of a pixel is measured using an energy function.

- **High energy:** edges, contours, textures
- **Low energy:** sky, walls, flat regions

### Energy Computation

The energy is computed from the image gradient:

```
E(x,y) = |∂I/∂x| + |∂I/∂y|
```

In practice, the derivatives are approximated using finite differences.

**Figure 1:** Energy map obtained using image gradients for a bird image

**Figure 2:** Energy map obtained using image gradients for a giraffe image

---

## Graph Representation

The image is modeled as a directed acyclic graph (DAG).

### Graph Construction

- Each pixel (x,y) represents a node.
- Directed edges connect a pixel to its three neighbors in the next row: (x-1, y+1), (x, y+1), (x+1, y+1)
- The weight of each edge is the energy of the destination pixel.

### Acyclic Property

Edges only go from row *y* to row *y+1*, therefore cycles are impossible. The graph is naturally a DAG.

---

## Shortest Path Formulation

Finding the optimal seam is equivalent to finding the shortest path from the top row to the bottom row of the DAG.

Because the graph is acyclic, the shortest path can be computed efficiently using dynamic programming following the topological order of rows.

---

## Dynamic Programming Approach

Let M(x,y) be the minimum cumulative energy required to reach pixel (x,y).

```
M(x,y) = E(x,y) + min(M(x-1, y-1), M(x, y-1), M(x+1, y-1))
```

### Algorithm Steps

1. Compute the energy map.
2. Initialize the first row of M.
3. Fill the matrix row by row.
4. Find the minimum value in the last row.
5. Backtrack to extract the seam.

### Code Extract

```python
# Dynamic programming computation
M[y, x] = energy[y, x] + min(
    M[y-1, x-1],
    M[y-1, x],
    M[y-1, x+1]
)
```

---

## Seam Extraction

Once the cumulative matrix is computed, the seam is obtained by backtracking from the minimum-energy pixel in the last row.

**Figure 3:** Minimal-energy seam highlighted in red of the bird image

**Figure 4:** Minimal-energy seam highlighted in red of the giraffe image

---

## Seam Removal

After identifying the seam, one pixel per row is removed. This reduces the image width by exactly one pixel.

**Figure 5:** Before and after seam carving of a bird image
- Original image (left)
- After seam carving (right)

**Figure 6:** Before and after seam carving of a giraffe image
- Original image (left)
- After seam carving (right)

The process can be repeated multiple times to achieve the desired width.

---

## Complexity Analysis

For removing a single seam:

- Energy computation: O(HW)
- Dynamic programming: O(HW)
- Seam removal: O(HW)

Removing *k* seams has complexity:

```
O(kHW)
```

Memory complexity is:

```
O(HW)
```

---

## Results and Discussion

The results demonstrate that Seam Carving preserves important visual content such as objects and edges, unlike classical scaling methods.

The algorithm effectively removes pixels from low-energy regions, allowing content-aware resizing.

---

## Limitations and Improvements

**Limitations:**
- Repeated seam removal may introduce artifacts.
- Computational cost is high for large images.

**Possible improvements:**
- Forward Energy computation
- Face detection and protection masks

---

## Conclusion

Seam Carving is a powerful example of how graph theory concepts can be applied to image processing.

By modeling an image as a directed acyclic graph and computing the shortest path, the algorithm resizes images while preserving important content.

This project highlights the strong relationship between:

- Graph theory
- Dynamic programming
- Computer vision

---

## References

- Avidan, S., Shamir, A. (2007). Seam Carving for Content-Aware Image Resizing.
- Adobe Photoshop Documentation.
