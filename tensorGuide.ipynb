{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "tensor (1).ipynb의 사본",
      "provenance": [],
      "collapsed_sections": [
        "Tce3stUlHN0L"
      ],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.5"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Vest1ge/Tensor/blob/main/tensorGuide.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "a74RapzxBuZM"
      },
      "source": [
        "이 포스트는 tensorflow 홈페이지 https://www.tensorflow.org/guide/tensor 내용을 번역하여 정리한 것 입니다. \n",
        "\n",
        "참고는 하되, 영문 내용을 통해 이해하는 것이 더욱 좋습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qFdPvlXBOdUN"
      },
      "source": [
        "# 텐서 소개"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:49.574367Z",
          "iopub.status.busy": "2021-09-22T20:44:49.573604Z",
          "iopub.status.idle": "2021-09-22T20:44:51.279550Z",
          "shell.execute_reply": "2021-09-22T20:44:51.278864Z"
        },
        "id": "AL2hzxorJiWy"
      },
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VQ3s2J8Vgowq"
      },
      "source": [
        "텐서는 `dtype` 이라고 하는 균일한 유형을 가진 다차원 배열입니다. `tf.dtypes.DType`에서 지원되는 모든 `dtype`을 볼 수 있습니다.\n",
        "\n",
        "만약 [NumPy](https://numpy.org/devdocs/user/quickstart.html)에 익숙하시다면 텐서는 일종의 `np.arrays`와 비슷하다고 생각하면 됩니다.\n",
        "\n",
        "모든 텐서는 Python의 숫자 및 문자열과 같이 불변합니다. 텐서의 내용을 업데이트할 수 없으며 오직 새로운 텐서만 생성할 수 있습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DRK5-9EpYbzG"
      },
      "source": [
        "## 기초\n",
        "\n",
        "먼저 기본적인 텐서를 한번 만들어 봅시다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uSHRFT6LJbxq"
      },
      "source": [
        "밑에 보이는 것은 **\"scalar\"** 또는 **\"rank-0\"** 텐서 입니다. 스칼라(scalar)는 단일 값을 포함하며 \"축(axes)\"은 포함하지 않습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TzPcZFs3GAGq"
      },
      "source": [
        "**※ 기본적으로 텐서플로우에 값을 할당하기 위해서는 `constant`라는 키워드를 사용합니다.**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.926698Z",
          "iopub.status.busy": "2021-09-22T20:44:52.925762Z",
          "iopub.status.idle": "2021-09-22T20:44:52.931951Z",
          "shell.execute_reply": "2021-09-22T20:44:52.932410Z"
        },
        "id": "d5JcgLFR6gHv"
      },
      "source": [
        "# 이 텐서는 기본적으로 int32 텐서가 됩니다. 아래 \"dtype\"을 참조하세요.\n",
        "rank_0_tensor = tf.constant(4) # 단일 값 4를 갖는 텐서"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a2JdWDOGFuUW",
        "outputId": "43c67fa3-43e6-48bc-87ab-60fd0006bccc"
      },
      "source": [
        "rank_0_tensor"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(), dtype=int32, numpy=4>"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tdmPAn9fWYs5"
      },
      "source": [
        "**\"vector\"** 또는 **\"rank-1\"** 텐서는 값(values)의 리스트(list)와 같습니다. 벡터(vector)는 하나의 축을 가집니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.939460Z",
          "iopub.status.busy": "2021-09-22T20:44:52.938607Z",
          "iopub.status.idle": "2021-09-22T20:44:52.941525Z",
          "shell.execute_reply": "2021-09-22T20:44:52.941943Z"
        },
        "id": "oZos8o_R6oE7"
      },
      "source": [
        "# 이번에는 \"dtype\"이 float인 텐서를 만들어 봅시다.\n",
        "rank_1_tensor = tf.constant([2.0, 3.0, 4.0]) # 소수점 뒤 0은 생략이 가능합니다."
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uOkC_bibHsyP",
        "outputId": "71ac9b45-eb46-4427-9465-1ce9f8af0f8b"
      },
      "source": [
        "rank_1_tensor"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(3,), dtype=float32, numpy=array([2., 3., 4.], dtype=float32)>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "G3IJG-ug_H4u"
      },
      "source": [
        "**\"matrix\"** 또는 **\"rank-2\"** 텐서는 두개의 축을 가집니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.947920Z",
          "iopub.status.busy": "2021-09-22T20:44:52.946897Z",
          "iopub.status.idle": "2021-09-22T20:44:52.950753Z",
          "shell.execute_reply": "2021-09-22T20:44:52.950262Z"
        },
        "id": "cnOIA_xb6u0M"
      },
      "source": [
        "# type을 명확하게 하고싶다면, 작성 시 \"dtype\"을 설정할 수도 있습니다. (아래 코드 참조)\n",
        "rank_2_tensor = tf.constant([[1, 2],\n",
        "                             [3, 4],\n",
        "                             [5, 6]], dtype=tf.float16) # \"dtype\"을 \"float16\"으로 설정"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7wGpuIu_JFgG",
        "outputId": "e49b3c09-6144-4ddd-a6d1-db4b7b46a99c"
      },
      "source": [
        "rank_2_tensor"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(3, 2), dtype=float16, numpy=\n",
              "array([[1., 2.],\n",
              "       [3., 4.],\n",
              "       [5., 6.]], dtype=float16)>"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iqCUpZCuKhFp"
      },
      "source": [
        "밑에 개념을 알기 쉽게 사진으로 설명되어 있습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "19m72qEPkfxi"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th>scalar, shape: <code>[]</code></th>\n",
        "  <th>vector, shape: <code>[3]</code></th>\n",
        "  <th>matrix, shape: <code>[3, 2]</code></th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "   <img src=\"img/scalar.png\">\n",
        "  </td>\n",
        "\n",
        "  <td>\n",
        "   <img src=\"img/vector.png\">\n",
        "  </td>\n",
        "  <td>\n",
        "   <img src=\"img/matrix.png\" alt=\"각 셀에 숫자가 포함된 3x2 그리드.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fjFvzcn4_ehD"
      },
      "source": [
        "텐서는 더 많은 축을 가질 수 있습니다. 다음은 세 개의 축을 가진 텐서입니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.957407Z",
          "iopub.status.busy": "2021-09-22T20:44:52.956693Z",
          "iopub.status.idle": "2021-09-22T20:44:52.959854Z",
          "shell.execute_reply": "2021-09-22T20:44:52.959223Z"
        },
        "id": "sesW7gw6JkXy"
      },
      "source": [
        "# 밑의 코드처럼 임의적으로 지정한 축이 있을 수 있습니다.(축은 차원이라고도 합니다).\n",
        "rank_3_tensor = tf.constant([\n",
        "                             [[0, 1, 2, 3, 4],\n",
        "                              [5, 6, 7, 8, 9]],\n",
        "                             [[10, 11, 12, 13, 14],\n",
        "                              [15, 16, 17, 18, 19]],\n",
        "                             [[20, 21, 22, 23, 24],\n",
        "                              [25, 26, 27, 28, 29]]])"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6Pa5_8RBMETK",
        "outputId": "0c84b826-d5e0-4ad5-cb75-944a130e4a7f"
      },
      "source": [
        "rank_3_tensor"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(3, 2, 5), dtype=int32, numpy=\n",
              "array([[[ 0,  1,  2,  3,  4],\n",
              "        [ 5,  6,  7,  8,  9]],\n",
              "\n",
              "       [[10, 11, 12, 13, 14],\n",
              "        [15, 16, 17, 18, 19]],\n",
              "\n",
              "       [[20, 21, 22, 23, 24],\n",
              "        [25, 26, 27, 28, 29]]], dtype=int32)>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rM2sTGIkoE3S"
      },
      "source": [
        "두 개 이상의 축을 가지고 있는 텐서를 시각화하는 방법에는 여러 가지 방법이 있습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NFiYfNMMhDgL"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th colspan=3>3축 텐서, shape: <code>[3, 2, 5]</code></th>\n",
        "<tr>\n",
        "<tr>\n",
        "  <td>\n",
        "   <img src=\"img/3-axis_numpy.png\"/>\n",
        "  </td>\n",
        "  <td>\n",
        "   <img src=\"img/3-axis_front.png\"/>\n",
        "  </td>\n",
        "\n",
        "  <td>\n",
        "   <img src=\"img/3-axis_block.png\"/>\n",
        "  </td>\n",
        "</tr>\n",
        "\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oWAc0U8OZwNb"
      },
      "source": [
        "`np.array` 또는 `tensor.numpy` 메소드를 사용하면 Tensorflow의 배열을 NumPy 배열로 변환할 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.972120Z",
          "iopub.status.busy": "2021-09-22T20:44:52.971269Z",
          "iopub.status.idle": "2021-09-22T20:44:52.974733Z",
          "shell.execute_reply": "2021-09-22T20:44:52.975163Z"
        },
        "id": "J5u6_6ZYaS7B",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8b3dac10-0290-40cd-ce0a-fab1dd299a7f"
      },
      "source": [
        "np.array(rank_2_tensor)"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1., 2.],\n",
              "       [3., 4.],\n",
              "       [5., 6.]], dtype=float16)"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.981351Z",
          "iopub.status.busy": "2021-09-22T20:44:52.980438Z",
          "iopub.status.idle": "2021-09-22T20:44:52.983490Z",
          "shell.execute_reply": "2021-09-22T20:44:52.983881Z"
        },
        "id": "c6Taz2gIaZeo",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2eadb2f8-67bc-4dd7-db79-df7eddede6fd"
      },
      "source": [
        "rank_2_tensor.numpy()"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1., 2.],\n",
              "       [3., 4.],\n",
              "       [5., 6.]], dtype=float16)"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hnz19F0ocEKD"
      },
      "source": [
        "텐서는 거의 대부분이 float와 int 타입이지만, 다음을 포함한 다른 타입도 존재합니다.\n",
        "\n",
        "* 복소수 (complex numbers)\n",
        "* 문자열 (strings)\n",
        "\n",
        "기본 `tf.Tensor` 클래스에서는 텐서가 \"직사각형\",\n",
        "\n",
        "즉, 각 축을 따라 모든 원소의 크기가 동일해야 합니다.\n",
        "\n",
        "그러나 다양한 형태를 처리할 수 있는 특수한 타입의 텐서도 존재합니다.\n",
        "\n",
        "* 비정형 텐서(Ragged tensors) ([링크참조](https://www.tensorflow.org/guide/ragged_tensor))\n",
        "* 희소 텐서(Sparse tensors) ([링크참조](https://www.tensorflow.org/guide/sparse_tensor))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SDC7OGeAIJr8"
      },
      "source": [
        "Tensorflow를 통해 덧셈, 원소별 곱셈 및 행렬 곱셈을 포함하여 텐서에 대한 기본적인 계산을 수행할 수 있습니다.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.990102Z",
          "iopub.status.busy": "2021-09-22T20:44:52.989188Z",
          "iopub.status.idle": "2021-09-22T20:44:52.993413Z",
          "shell.execute_reply": "2021-09-22T20:44:52.993809Z"
        },
        "id": "-DTkjwDOIIDa",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1799e01e-94de-4d5c-986f-2cf23223d677"
      },
      "source": [
        "a = tf.constant([[1, 2],\n",
        "                 [3, 4]])\n",
        "b = tf.constant([[1, 1],\n",
        "                 [1, 1]]) \n",
        "# `tf.ones([2,2])` 코드를 이용해도 b와 똑같은 행렬이 나온다. (원소 값이 모두 1인 2x2 행렬)\n",
        "\n",
        "print(tf.add(a, b), \"\\n\") # 원소별 덧셈\n",
        "print(tf.multiply(a, b), \"\\n\") # 원소별 곱셈\n",
        "print(tf.matmul(a, b), \"\\n\") # 행렬 곱셈"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[2 3]\n",
            " [4 5]], shape=(2, 2), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[1 2]\n",
            " [3 4]], shape=(2, 2), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[3 3]\n",
            " [7 7]], shape=(2, 2), dtype=int32) \n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "y85CO6ZeVOuN"
      },
      "source": [
        "이 방법을 사용할 수도 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:52.999015Z",
          "iopub.status.busy": "2021-09-22T20:44:52.998022Z",
          "iopub.status.idle": "2021-09-22T20:44:53.002513Z",
          "shell.execute_reply": "2021-09-22T20:44:53.002028Z"
        },
        "id": "2smoWeUz-N2q",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b8a957d6-1a52-4bb4-999a-a31acbf967fd"
      },
      "source": [
        "print(a + b, \"\\n\") # 원소별 덧셈\n",
        "print(a * b, \"\\n\")  # 원소별 곱셈\n",
        "print(a @ b, \"\\n\") # 행렬 곱셈"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[2 3]\n",
            " [4 5]], shape=(2, 2), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[1 2]\n",
            " [3 4]], shape=(2, 2), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[3 3]\n",
            " [7 7]], shape=(2, 2), dtype=int32) \n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S3_vIAl2JPVc"
      },
      "source": [
        "텐서는 모든 종류의 연산(ops)작업에 사용됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.008419Z",
          "iopub.status.busy": "2021-09-22T20:44:53.007581Z",
          "iopub.status.idle": "2021-09-22T20:44:53.012495Z",
          "shell.execute_reply": "2021-09-22T20:44:53.012016Z"
        },
        "id": "Gp4WUYzGIbnv",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4858de16-c7a1-4312-c6a9-60560de62d31"
      },
      "source": [
        "c = tf.constant([[4.0, 5.0], [10.0, 1.0]])\n",
        "\n",
        "# 가장 큰 값 찾기\n",
        "print(tf.reduce_max(c))\n",
        "# 가장 큰 값의 인덱스 찾기\n",
        "print(tf.argmax(c))\n",
        "# 소프트맥스 계산\n",
        "print(tf.nn.softmax(c))"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(10.0, shape=(), dtype=float32)\n",
            "tf.Tensor([1 0], shape=(2,), dtype=int64)\n",
            "tf.Tensor(\n",
            "[[2.6894143e-01 7.3105854e-01]\n",
            " [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NvSAbowVVuRr"
      },
      "source": [
        "## 형상(shapes) 에 대하여"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hkaBIqkTCcGY"
      },
      "source": [
        "텐서는 **shapes**를 가지고 있습니다. 사용되는 일부 용어는 다음과 같습니다.\n",
        "\n",
        "* **형상(Shape)**: 텐서의 각 축의 길이(원소의 수)입니다.\n",
        "* **순위(Rank)**: 텐서 축의 수입니다.  \n",
        " 예시) rank가 0인 \"scalar\", rank가 1인 \"vector\", rank가 2인 \"matrix\".\n",
        "* **축(Axis)** 또는 **차원(Dimension)**: 텐서의 특정한 차원\n",
        "* **크기(Size)**: 텐서의 총 항목 수. 곱 형상의 벡터\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "E9L3-kCQq2f6"
      },
      "source": [
        "참고: \"2차원 텐서\"에 대한 참조가 있을 수 있지만, rank-2 텐서는 일반적으로 2차원 공간을 설명하지 않습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VFOyG2tn8LhW"
      },
      "source": [
        "텐서와 `tf.TensorShape` 개체는 다음 항목에 액세스할 수 있는 편리한 속성을 가지고 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.017885Z",
          "iopub.status.busy": "2021-09-22T20:44:53.017226Z",
          "iopub.status.idle": "2021-09-22T20:44:53.019375Z",
          "shell.execute_reply": "2021-09-22T20:44:53.019760Z"
        },
        "id": "RyD3yewUKdnK"
      },
      "source": [
        "rank_4_tensor = tf.zeros([3, 2, 4, 5])"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oTZZW9ziq4og"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th colspan=2>rank-4 텐서, shape: <code>[3, 2, 4, 5]</code></th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/shape.png\" alt=\"A tensor shape is like a vector.\">\n",
        "    <td>\n",
        "<img src=\"img/4-axis_block.png\" alt=\"A 4-axis tensor\">\n",
        "  </td>\n",
        "  </tr>\n",
        "</table>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.025892Z",
          "iopub.status.busy": "2021-09-22T20:44:53.024709Z",
          "iopub.status.idle": "2021-09-22T20:44:53.030015Z",
          "shell.execute_reply": "2021-09-22T20:44:53.029510Z"
        },
        "id": "MHm9vSqogsBk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1cb28a76-50c9-4040-9de3-a7e41e8bd60b"
      },
      "source": [
        "print(\"모든 원소의 타입:\", rank_4_tensor.dtype)\n",
        "print(\"축(차원)의 수:\", rank_4_tensor.ndim)\n",
        "print(\"텐서의 형상(shape):\", rank_4_tensor.shape)\n",
        "print(\"축(차원)0을 따르는 원소:\", rank_4_tensor.shape[0])\n",
        "print(\"마지막 축(차원)을 따르는 원소:\", rank_4_tensor.shape[-1])\n",
        "print(\"총 원소의 수(3*2*4*5): \", tf.size(rank_4_tensor).numpy())"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "모든 원소의 타입: <dtype: 'float32'>\n",
            "축(차원)의 수: 4\n",
            "텐서의 형상(shape): (3, 2, 4, 5)\n",
            "축(차원)0을 따르는 원소: 3\n",
            "마지막 축(차원)을 따르는 원소: 5\n",
            "총 원소의 수(3*2*4*5):  120\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bQmE_Vx5JilS"
      },
      "source": [
        "While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.\n",
        "\n",
        "축은 종종 인덱스로 참조되지만 각 축의 의미를 항상 추적해야 합니다. 축은 대개 전역에서 로컬 순서로 정렬됩니다. 배치 축에 이어 공간 차원 및 각 위치의 특성이 맨 마지막에 옵니다. 이러한 방식으로 특성 벡터(feature vectors)는 메모리의 연속적인 영역입니다.\n",
        "\n",
        "<table>\n",
        "<tr>\n",
        "<th>일반적인 축 순서</th>\n",
        "</tr>\n",
        "<tr>\n",
        "    <td>\n",
        "<img src=\"img/shape2.png\" alt=\"각 축이 무엇인지 추적합니다. 4축 텐서는 배치, 너비, 높이, 특징일 수 있습니다.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FlPoVvJS75Bb"
      },
      "source": [
        "## 인덱싱"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "apOkCKqCZIZu"
      },
      "source": [
        "### 단일 축 인덱싱(Single-axis indexing)\n",
        "\n",
        "Tensorflow는 [파이썬의 목록이나 문자열을 인덱싱하는 것](https://docs.python.org/3/tutorial/introduction.html#strings)과 유사한 표준 파이썬 인덱싱 규칙과 NumPy 인덱싱의 기본 규칙을 따릅니다.\n",
        "* 인덱스는 `0`에서 부터 시작합니다.\n",
        "* 음수 인덱스는 끝에서부터 거꾸로 계산합니다.\n",
        "* 콜론 `:`은 슬라이스에 사용됩니다. `start:stop:step`\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.036406Z",
          "iopub.status.busy": "2021-09-22T20:44:53.035730Z",
          "iopub.status.idle": "2021-09-22T20:44:53.038227Z",
          "shell.execute_reply": "2021-09-22T20:44:53.038713Z"
        },
        "id": "SQ-CrJxLXTIM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e3180b1e-3939-473b-eaff-f36ca0e0b102"
      },
      "source": [
        "rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])\n",
        "rank_1_tensor.numpy()"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34], dtype=int32)"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mQYYL56PXSak"
      },
      "source": [
        "스칼라를 사용하여 인덱싱하면 축이 제거됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.044592Z",
          "iopub.status.busy": "2021-09-22T20:44:53.043681Z",
          "iopub.status.idle": "2021-09-22T20:44:53.048738Z",
          "shell.execute_reply": "2021-09-22T20:44:53.048185Z"
        },
        "id": "n6tqHciOWMt5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "32c05d6e-333e-4e44-91d0-c7097acf637f"
      },
      "source": [
        "print(\"0번 인덱스(1번째 원소):\", rank_1_tensor[0].numpy())\n",
        "print(\"1번 인덱스(2번째 원소):\", rank_1_tensor[1].numpy())\n",
        "print(\"마지막 원소:\", rank_1_tensor[-1].numpy())"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0번 인덱스(1번째 원소): 0\n",
            "1번 인덱스(2번째 원소): 1\n",
            "마지막 원소: 34\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qJLHU_a2XwpG"
      },
      "source": [
        "`:` 슬라이스를 사용하여 인덱싱하면 축이 유지 됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.054856Z",
          "iopub.status.busy": "2021-09-22T20:44:53.053639Z",
          "iopub.status.idle": "2021-09-22T20:44:53.061369Z",
          "shell.execute_reply": "2021-09-22T20:44:53.060861Z"
        },
        "id": "giVPPcfQX-cu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f9a9897e-3483-44e7-ed58-0eec235d7e14"
      },
      "source": [
        "print(\"모든 원소:\", rank_1_tensor[:].numpy())\n",
        "print(\"4번 인덱스 전 모든 원소:\", rank_1_tensor[:4].numpy())\n",
        "print(\"4번 인덱스부터 끝까지:\", rank_1_tensor[4:].numpy())\n",
        "print(\"2번 인덱스부터 7번 인덱스까지:\", rank_1_tensor[2:7].numpy())\n",
        "print(\"2 인덱스씩 건너뛰며:\", rank_1_tensor[::2].numpy())\n",
        "print(\"원소 거꾸로 정렬:\", rank_1_tensor[::-1].numpy())"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "모든 원소: [ 0  1  1  2  3  5  8 13 21 34]\n",
            "4번 인덱스 전 모든 원소: [0 1 1 2]\n",
            "4번 인덱스부터 끝까지: [ 3  5  8 13 21 34]\n",
            "2번 인덱스부터 7번 인덱스까지: [1 2 3 5 8]\n",
            "2 인덱스씩 건너뛰며: [ 0  1  3  8 21]\n",
            "원소 거꾸로 정렬: [34 21 13  8  5  3  2  1  1  0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "elDSxXi7X-Bh"
      },
      "source": [
        "### 다차원 인덱싱(Multi-axis indexing)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Cgk0uRUYZiai"
      },
      "source": [
        "상위 텐서는 여러 인덱스를 전달하여 인덱싱됩니다.\n",
        "\n",
        "단일 축 사례와 정확히 동일한 규칙이 각 축에 독립적으로 적용됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.067199Z",
          "iopub.status.busy": "2021-09-22T20:44:53.066389Z",
          "iopub.status.idle": "2021-09-22T20:44:53.069106Z",
          "shell.execute_reply": "2021-09-22T20:44:53.069550Z"
        },
        "id": "Tc5X_WlsZXmd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3bbc1c4b-6036-40b0-b76f-46153edc8cd2"
      },
      "source": [
        "print(rank_2_tensor.numpy())"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1. 2.]\n",
            " [3. 4.]\n",
            " [5. 6.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "w07U9vq5ipQk"
      },
      "source": [
        "각 인덱스에 대해 정수를 전달하면 결과는 스칼라(scalar)로 나옵니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.074831Z",
          "iopub.status.busy": "2021-09-22T20:44:53.074140Z",
          "iopub.status.idle": "2021-09-22T20:44:53.076938Z",
          "shell.execute_reply": "2021-09-22T20:44:53.077365Z"
        },
        "id": "PvILXc1PjqTM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8b9cb654-451a-4038-8505-be5bd5e6d611"
      },
      "source": [
        "# rank-2 텐서에서 단일 값을 추출\n",
        "print(rank_2_tensor[1, 1].numpy())"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "4.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3RLCzAOHjfEH"
      },
      "source": [
        "다음과 같은 정수 및 `:`슬라이스의 조합을 사용하여 인덱싱할 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.083796Z",
          "iopub.status.busy": "2021-09-22T20:44:53.083008Z",
          "iopub.status.idle": "2021-09-22T20:44:53.089415Z",
          "shell.execute_reply": "2021-09-22T20:44:53.089836Z"
        },
        "id": "YTqNqsfJkJP_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "736216de-56a2-43cf-aec0-3f9295059d8b"
      },
      "source": [
        "# 행 과 열 텐서 가져오기\n",
        "print(\"두번째 행:\", rank_2_tensor[1, :].numpy())\n",
        "print(\"두번째 열:\", rank_2_tensor[:, 1].numpy())\n",
        "print(\"마지막 행:\", rank_2_tensor[-1, :].numpy())\n",
        "print(\"마지막 열의 첫번째 원소:\", rank_2_tensor[0, -1].numpy())\n",
        "print(\"첫번째 열 제외:\")\n",
        "print(rank_2_tensor[1:, :].numpy(), \"\\n\")"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "두번째 행: [3. 4.]\n",
            "두번째 열: [2. 4. 6.]\n",
            "마지막 행: [5. 6.]\n",
            "마지막 열의 첫번째 원소: 2.0\n",
            "첫번째 열 제외:\n",
            "[[3. 4.]\n",
            " [5. 6.]] \n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "P45TwSUVSK6G"
      },
      "source": [
        "3차원 텐서의 예는 다음과 같다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.094272Z",
          "iopub.status.busy": "2021-09-22T20:44:53.093664Z",
          "iopub.status.idle": "2021-09-22T20:44:53.096447Z",
          "shell.execute_reply": "2021-09-22T20:44:53.096805Z"
        },
        "id": "GuLoMoCVSLxK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d0f908a2-3bd1-4cf3-a5bf-cb38905112cc"
      },
      "source": [
        "print(rank_3_tensor[:, :, 4])"
      ],
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[ 4  9]\n",
            " [14 19]\n",
            " [24 29]], shape=(3, 2), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9NgmHq27TJOE"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "<th colspan=2>배치에서 각 예의 모든 위치에서 마지막 특성 선택하기 </th>\n",
        "</tr>\n",
        "<tr>\n",
        "    <td>\n",
        "<img src=\"img/index1.png\" alt=\"마지막 축의 인덱스-4에서 모든 값이 선택된 3x2x5 텐서.\">\n",
        "  </td>\n",
        "      <td>\n",
        "<img src=\"img/tensor/index2.png\" alt=\"선택한 값은 2축 텐서에 패키지된다.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t9V83-thHn89"
      },
      "source": [
        "[텐서 슬라이싱 가이드](https://tensorflow.org/guide/tensor_slicing)를 읽고 인덱싱을 적용하여 텐서의 개별 원소를 조작하는 방법에 대해 알아본다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fpr7R0t4SVb0"
      },
      "source": [
        "## 형상(Shapes) 조작하기\n",
        "\n",
        "텐서의 형상을 바꾸는 것은 매우 유용합니다.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.101718Z",
          "iopub.status.busy": "2021-09-22T20:44:53.101170Z",
          "iopub.status.idle": "2021-09-22T20:44:53.103532Z",
          "shell.execute_reply": "2021-09-22T20:44:53.102963Z"
        },
        "id": "EMeTtga5Wq8j",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "29b61483-a171-4957-9a87-c0c1a0e377bb"
      },
      "source": [
        "# 형상(Shape)는 각 축의 크기를 표시하는 'TensorShape' 개체를 반환합니다.\n",
        "x = tf.constant([[1], [2], [3]])\n",
        "print(x.shape)"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(3, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.107468Z",
          "iopub.status.busy": "2021-09-22T20:44:53.106710Z",
          "iopub.status.idle": "2021-09-22T20:44:53.109817Z",
          "shell.execute_reply": "2021-09-22T20:44:53.109310Z"
        },
        "id": "38jc2RXziT3W",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b22c0daa-ccb3-4861-8a75-8fbd3369b8f3"
      },
      "source": [
        "# 이 개체를 파이썬 리스트로 변환할 수도 있습니다.\n",
        "print(x.shape.as_list())"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[3, 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "J_xRlHZMKYnF"
      },
      "source": [
        "텐서를 새 형상으로 재구성할 수 있습니다. `tf.reshape` 는 기본 데이터를 복제할 필요가 없어 재구성이 빠릅니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.114448Z",
          "iopub.status.busy": "2021-09-22T20:44:53.113843Z",
          "iopub.status.idle": "2021-09-22T20:44:53.115902Z",
          "shell.execute_reply": "2021-09-22T20:44:53.116266Z"
        },
        "id": "pa9JCgMLWy87"
      },
      "source": [
        "# 텐서를 새 형상으로 재구성할 수 있습니다.\n",
        "# 리스트를 전달한다는 점에 유의하세요.\n",
        "reshaped = tf.reshape(x, [1, 3])"
      ],
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.120908Z",
          "iopub.status.busy": "2021-09-22T20:44:53.120196Z",
          "iopub.status.idle": "2021-09-22T20:44:53.122994Z",
          "shell.execute_reply": "2021-09-22T20:44:53.122406Z"
        },
        "id": "Mcq7iXOkW3LK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9fe68f14-aa8a-43d3-97f4-e564a048c3a0"
      },
      "source": [
        "print(x.shape)\n",
        "print(reshaped.shape)"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(3, 1)\n",
            "(1, 3)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gIB2tOkoVr6E"
      },
      "source": [
        "데이터는 메모리에 레이아웃을 유지하고 요청한 형상이 동일한 데이터를 가리키는 새로운 텐서가 생성됩니다. TensorFlow는 C 스타일의 \"행 중심\" 메모리 순서를 사용하며, 여기서 가장 오른쪽 인덱스를 증가시키는 것은 메모리의 단일 단계에 해당합니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.128067Z",
          "iopub.status.busy": "2021-09-22T20:44:53.127422Z",
          "iopub.status.idle": "2021-09-22T20:44:53.129639Z",
          "shell.execute_reply": "2021-09-22T20:44:53.130065Z"
        },
        "id": "7kMfM0RpUgI8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c2fea493-6e60-419f-9853-ce1c82dd431d"
      },
      "source": [
        "print(rank_3_tensor)"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[[ 0  1  2  3  4]\n",
            "  [ 5  6  7  8  9]]\n",
            "\n",
            " [[10 11 12 13 14]\n",
            "  [15 16 17 18 19]]\n",
            "\n",
            " [[20 21 22 23 24]\n",
            "  [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TcDtfQkJWzIx"
      },
      "source": [
        "텐서의 평탄화 하면 메모리에 배열된 순서를 알 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.135471Z",
          "iopub.status.busy": "2021-09-22T20:44:53.134802Z",
          "iopub.status.idle": "2021-09-22T20:44:53.137182Z",
          "shell.execute_reply": "2021-09-22T20:44:53.137613Z"
        },
        "id": "COnHEPuaWDQp",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a7a13eaf-b50d-4e72-cfee-a16cd7c00b64"
      },
      "source": [
        "# 특수 값 -1이면 전체 크기가 일정하게 유지되도록 해당 차원의 크기가 계산됩니다.\n",
        "print(tf.reshape(rank_3_tensor, [-1]))"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n",
            " 24 25 26 27 28 29], shape=(30,), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jJZRira2W--c"
      },
      "source": [
        "Typically the only reasonable use of `tf.reshape` is to combine or split adjacent axes (or add/remove `1`s).\n",
        "\n",
        "일반적으로 `tf.reshape`의 합리적인 용도는 인접한 축을 결합하거나 분할하는 것(또는 `1`을 추가/제거하는 것)뿐이다.\n",
        "\n",
        "For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do, as the slices do not mix:\n",
        "\n",
        "이 3x2x5 텐서의 경우 (3x2)x5 또는 3x(2x5)로 재구성하는 것이 슬라이스가 섞이지 않기 때문에 합리적인 재구성 방법이라고 할 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.144946Z",
          "iopub.status.busy": "2021-09-22T20:44:53.143743Z",
          "iopub.status.idle": "2021-09-22T20:44:53.147447Z",
          "shell.execute_reply": "2021-09-22T20:44:53.147869Z"
        },
        "id": "zP2Iqc7zWu_J",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "14d37dde-5853-43a3-e8e9-5d1fbf105f88"
      },
      "source": [
        "print(tf.reshape(rank_3_tensor, [3*2, 5]), \"\\n\")\n",
        "print(tf.reshape(rank_3_tensor, [3, -1]))"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[ 0  1  2  3  4]\n",
            " [ 5  6  7  8  9]\n",
            " [10 11 12 13 14]\n",
            " [15 16 17 18 19]\n",
            " [20 21 22 23 24]\n",
            " [25 26 27 28 29]], shape=(6, 5), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[ 0  1  2  3  4  5  6  7  8  9]\n",
            " [10 11 12 13 14 15 16 17 18 19]\n",
            " [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6ZsZRUhihlDB"
      },
      "source": [
        "<table>\n",
        "<th colspan=3>\n",
        "몇 가지 좋은 재구성\n",
        "</th>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/reshape-before.png\" alt=\"3x2x5 텐서\">\n",
        "  </td>\n",
        "  <td>\n",
        "  <img src=\"img/reshape-good1.png\" alt=\"(3x2)x5로 재구성된 동일한 데이터\">\n",
        "  </td>\n",
        "  <td>\n",
        "  <img src=\"img/reshape-good2.png\" alt=\"3x(2x5)로 재구성된 동일한 데이터\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nOcRxDC3jNIU"
      },
      "source": [
        "Reshaping will \"work\" for any new shape with the same total number of elements, but it will not do anything useful if you do not respect the order of the axes.\n",
        "\n",
        "전체 원소 수가 동일한 새 형상에 대해 재구성되지만 축의 순서를 고려하지 않으면 유용하게 사용할 수 없습니다.\n",
        "\n",
        "`tf.reshape`에서 축 교환이 작동하지 않으면, `tf.transpose`를 수행하여야 합니다.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.154381Z",
          "iopub.status.busy": "2021-09-22T20:44:53.153667Z",
          "iopub.status.idle": "2021-09-22T20:44:53.157221Z",
          "shell.execute_reply": "2021-09-22T20:44:53.156708Z"
        },
        "id": "I9qDL_8u7cBH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "08bcdc9d-3cf7-4b60-8612-6649210f7a15"
      },
      "source": [
        "# 이것은 나쁜 예시입니다.\n",
        "\n",
        "# 형상을 재구성하면 축의 순서를 변경할 수 없습니다.\n",
        "print(tf.reshape(rank_3_tensor, [2, 3, 5]), \"\\n\") \n",
        "\n",
        "# 매우 지저분한 형상으로 재구성됩니다.\n",
        "print(tf.reshape(rank_3_tensor, [5, 6]), \"\\n\")\n",
        "\n",
        "# 전혀 효과가 없는 코드입니다.\n",
        "try:\n",
        "  tf.reshape(rank_3_tensor, [7, -1])\n",
        "except Exception as e:\n",
        "  print(f\"{type(e).__name__}: {e}\")"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[[ 0  1  2  3  4]\n",
            "  [ 5  6  7  8  9]\n",
            "  [10 11 12 13 14]]\n",
            "\n",
            " [[15 16 17 18 19]\n",
            "  [20 21 22 23 24]\n",
            "  [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[ 0  1  2  3  4  5]\n",
            " [ 6  7  8  9 10 11]\n",
            " [12 13 14 15 16 17]\n",
            " [18 19 20 21 22 23]\n",
            " [24 25 26 27 28 29]], shape=(5, 6), dtype=int32) \n",
            "\n",
            "InvalidArgumentError: Input to reshape is a tensor with 30 values, but the requested shape requires a multiple of 7 [Op:Reshape]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qTM9-5eh68oo"
      },
      "source": [
        "<table>\n",
        "<th colspan=3>\n",
        "몇가지 좋지 않은 재구성\n",
        "</th>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/reshape-bad.png\" alt=\"축의 순서를 변경할 수 없습니다. tf.transpose를 사용하십시오.\">\n",
        "  </td>\n",
        "  <td>\n",
        "<img src=\"img/reshape-bad4.png\" alt=\"Anything that mixes the slices of data together is probably wrong.\">\n",
        "  </td>\n",
        "  <td>\n",
        "<img src=\"img/reshape-bad2.png\" alt=\"새 형상이 정확하게 맞아야 합니다.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "N9r90BvHCbTt"
      },
      "source": [
        "완전히 지정되지 않은 형상에 대해서 실행할 수 있습니다. 형상에 `None`(축 길이를 알 수 없음)이 포함되거나 전체 형상에 `None`(텐서의 rank를 알 수 없음)이 포함되는 것을 말합니다.\n",
        "\n",
        "이러한 것은 [tf.RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)를 제외하고, TensorFlow의 상징적인 그래프 빌딩 API 컨텍스트에서만 발생합니다.\n",
        "\n",
        "* [tf.function](https://www.tensorflow.org/guide/function) \n",
        "* [keras functional API](https://www.tensorflow.org/guide/keras/functional).\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fDmFtFM7k0R2"
      },
      "source": [
        "## `DTypes`에 관한 추가 정보\n",
        "\n",
        "`tf.Tensor`의 데이터 타입을 검사하기 위해, `Tensor.dtype` 속성을 사용합니다.\n",
        "\n",
        "Python 객체에서 `tf.Tensor`를 만들 때 선택적으로 데이터 타입을 지정할 수 있습니다.\n",
        "\n",
        "지정을 하지 않으면, TensorFlow는 데이터를 나타낼 수 있는 데이터 타입을 선택합니다. TensorFlow는 Python 정수를 `tf.int32`로, Python 부동 소수점 숫자를 `tf.float32`로 변환합니다. TensorFlow는 NumPy가 배열로 변환할 때 사용하는 것과 같은 규칙을 사용합니다.\n",
        "\n",
        "유형별로 지정할 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.163021Z",
          "iopub.status.busy": "2021-09-22T20:44:53.162303Z",
          "iopub.status.idle": "2021-09-22T20:44:53.165702Z",
          "shell.execute_reply": "2021-09-22T20:44:53.166087Z"
        },
        "id": "5mSTDWbelUvu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "632ab030-451a-4e1e-ee17-c37721a9cc7f"
      },
      "source": [
        "the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)\n",
        "the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)\n",
        "# uint8로 지정하면 소수점의 부분을 잃습니다.\n",
        "the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)\n",
        "print(the_u8_tensor)"
      ],
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor([2 3 4], shape=(3,), dtype=uint8)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s1yBlJsVlFSu"
      },
      "source": [
        "## 브로드캐스팅(Broadcasting)\n",
        "\n",
        "브로드캐스팅은 [NumPy의 해당 기능](https://numpy.org/doc/stable/user/basics.html)에서 차용된 개념입니다. 즉, 특정 조건에서 작은 텐서가 결합 연산을 실행할 때 자동으로 \"확장(streched)\"되어 더 큰 텐서에 맞게되는 것을 말합니다.\n",
        "\n",
        "가장 간단하고 일반적인 경우는 스칼라(scalar)에 텐서를 곱하거나 추가하려고 할 때입니다. \n",
        "\n",
        "이 경우, 스칼라는 다른 인수와 동일한 형상으로 브로드캐스트됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.172143Z",
          "iopub.status.busy": "2021-09-22T20:44:53.171428Z",
          "iopub.status.idle": "2021-09-22T20:44:53.175904Z",
          "shell.execute_reply": "2021-09-22T20:44:53.175397Z"
        },
        "id": "P8sypqmagHQN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "09ecd74b-fb6b-4e0f-9fa1-6a289c81d7a3"
      },
      "source": [
        "x = tf.constant([1, 2, 3])\n",
        "\n",
        "y = tf.constant(2)\n",
        "z = tf.constant([2, 2, 2])\n",
        "# 밑에 있는 모든 연산의 결과가 같다.\n",
        "print(tf.multiply(x, 2))\n",
        "print(x * y)\n",
        "print(x * z)"
      ],
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor([2 4 6], shape=(3,), dtype=int32)\n",
            "tf.Tensor([2 4 6], shape=(3,), dtype=int32)\n",
            "tf.Tensor([2 4 6], shape=(3,), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "o0SBoR6voWcb"
      },
      "source": [
        "Likewise, axes with length 1 can be stretched out to match the other arguments.  Both arguments can be stretched in the same computation.\n",
        "\n",
        "마찬가지로 크기가 1인 축도 다른 인수와 일치하도록 확장할 수 있습니다. 두 인수 모두 동일한 계산으로 확장할 수 있습니다.\n",
        "\n",
        "In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to produce a 3x4 matrix. Note how the leading 1 is optional: The shape of y is `[4]`.\n",
        "\n",
        "이 경우, 3x1 행렬에 1x4 행렬을 원소별 곱셈하면 3x4 행렬이 생성됩니다. 선행 1이 선택 사항인 점에 유의하세요. y의 형상은 `[4]`입니다.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.182376Z",
          "iopub.status.busy": "2021-09-22T20:44:53.181687Z",
          "iopub.status.idle": "2021-09-22T20:44:53.185364Z",
          "shell.execute_reply": "2021-09-22T20:44:53.184795Z"
        },
        "id": "6sGmkPg3XANr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6376be77-c804-4db7-e551-2af381062a3d"
      },
      "source": [
        "# 이것들은 같은 연산이다.\n",
        "x = tf.reshape(x,[3,1])\n",
        "y = tf.range(1, 5)\n",
        "print(x, \"\\n\")\n",
        "print(y, \"\\n\")\n",
        "print(tf.multiply(x, y))"
      ],
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[1]\n",
            " [2]\n",
            " [3]], shape=(3, 1), dtype=int32) \n",
            "\n",
            "tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) \n",
            "\n",
            "tf.Tensor(\n",
            "[[ 1  2  3  4]\n",
            " [ 2  4  6  8]\n",
            " [ 3  6  9 12]], shape=(3, 4), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t_7sh-EUYLrE"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th>추가 시 브로드캐스팅: <code>[3, 1]</code> 와 <code>[1, 4]</code> 의 곱하기는 <code>[3, 4]</code> 입니다. </th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/broadcasting.png\" alt=\"4x1 행렬에 3x1 행렬을 추가하면 3x4 행렬이 생성됩니다.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9V3KgSJcKDRz"
      },
      "source": [
        "같은 연산이지만 브로드캐스팅이 없는 연산이 여기 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.191395Z",
          "iopub.status.busy": "2021-09-22T20:44:53.190745Z",
          "iopub.status.idle": "2021-09-22T20:44:53.192945Z",
          "shell.execute_reply": "2021-09-22T20:44:53.193338Z"
        },
        "id": "elrF6v63igY8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "01e42737-313b-4393-896c-ff03eddd6f08"
      },
      "source": [
        "x_stretch = tf.constant([[1, 1, 1, 1],\n",
        "                         [2, 2, 2, 2],\n",
        "                         [3, 3, 3, 3]])\n",
        "\n",
        "y_stretch = tf.constant([[1, 2, 3, 4],\n",
        "                         [1, 2, 3, 4],\n",
        "                         [1, 2, 3, 4]])\n",
        "\n",
        "print(x_stretch * y_stretch)  # 연산자를 다시 오버로딩"
      ],
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[ 1  2  3  4]\n",
            " [ 2  4  6  8]\n",
            " [ 3  6  9 12]], shape=(3, 4), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "14KobqYu85gi"
      },
      "source": [
        "브로드캐스팅은 브로드캐스트 연산으로 메모리의 확장된 텐서를 구체화하지 않기 때문에 대부분의 경우 시간과 공간적으로 모두 효율적입니다.\n",
        "\n",
        "`tf.broadcast_to`를 사용하면 어떤 모습을 하고있는지 알 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.197994Z",
          "iopub.status.busy": "2021-09-22T20:44:53.197363Z",
          "iopub.status.idle": "2021-09-22T20:44:53.200299Z",
          "shell.execute_reply": "2021-09-22T20:44:53.199872Z"
        },
        "id": "GW2Q59_r8hZ6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "51f97af5-559d-4853-de20-4f39467c3334"
      },
      "source": [
        "print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))"
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(\n",
            "[[1 2 3]\n",
            " [1 2 3]\n",
            " [1 2 3]], shape=(3, 3), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Z2bAMMQY-jpP"
      },
      "source": [
        "Unlike a mathematical op, for example, `broadcast_to` does nothing special to save memory.  Here, you are materializing the tensor.\n",
        "\n",
        "예를 들어, 수학적 연산과 달리 `broadcast_to`는 메모리를 절약하기 위해 특별한 연산을 수행하지 않습니다. 여기에서 텐서를 구체화해봅시다.\n",
        "\n",
        "훨씬 더 복잡해질 수 있습니다.  [해당 섹션](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) 에서는 더 많은 브로드캐스팅 트릭을 보여줍니다. (NumPy 에서)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "o4Rpz0xAsKSI"
      },
      "source": [
        "## tf.convert_to_tensor\n",
        "\n",
        "`tf.matmul` 및 `tf.reshape`와 같은 대부분의 ops는 클래스 `tf.Tensor`의 인수를 사용합니다. 그러나 위의 경우, 텐서 형상의 Python 객체가 수용됨을 알 수 있습니다.\n",
        "\n",
        "전부는 아니지만 대부분의 ops는 텐서가 아닌 인수에 대해 `convert_to_tensor`를 호출합니다. 변환 레지스트리가 있어 NumPy의 `ndarray`, `TensorShape` , Python 목록 및 `tf.Variable`과 같은 대부분의 객체 클래스는 모두 자동으로 변환됩니다.\n",
        "\n",
        "자세한 내용은 [`tf.register_tensor_conversion_function`](https://www.tensorflow.org/api_docs/python/tf/register_tensor_conversion_function)을 참조하세요. 자신만의 유형이 있으면 자동으로 텐서로 변환할 수 있습니다.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "05bBVBVYV0y6"
      },
      "source": [
        "## 비정형 텐서(Ragged Tensors)\n",
        "\n",
        "어떤 축을 따라 다양한 수의 요소를 가진 텐서를 \"비정형(ragged)\"이라고 합니다. 비정형 데이터에는 `tf.ragged.RaggedTensor`를 사용합니다.\n",
        "\n",
        "예를 들어, 비정형 텐서는 정규 텐서로 표현할 수 없습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VPc3jGoeJqB7"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th>A `tf.RaggedTensor`, shape: <code>[4, None]</code></th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/ragged.png\" alt=\"2축 비정형 텐서는 각 행의 길이가 다를 수 있습니다.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.204961Z",
          "iopub.status.busy": "2021-09-22T20:44:53.204321Z",
          "iopub.status.idle": "2021-09-22T20:44:53.206574Z",
          "shell.execute_reply": "2021-09-22T20:44:53.206953Z"
        },
        "id": "VsbTjoFfNVBF"
      },
      "source": [
        "ragged_list = [\n",
        "    [0, 1, 2, 3],\n",
        "    [4, 5],\n",
        "    [6, 7, 8],\n",
        "    [9]]"
      ],
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.211681Z",
          "iopub.status.busy": "2021-09-22T20:44:53.211072Z",
          "iopub.status.idle": "2021-09-22T20:44:53.213147Z",
          "shell.execute_reply": "2021-09-22T20:44:53.213555Z"
        },
        "id": "p4xKTo57tutG",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a1b2c0e3-dcdc-43e1-8d96-71160b771137"
      },
      "source": [
        "try:\n",
        "  tensor = tf.constant(ragged_list)\n",
        "except Exception as e:\n",
        "  print(f\"{type(e).__name__}: {e}\")"
      ],
      "execution_count": 59,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ValueError: Can't convert non-rectangular Python sequence to Tensor.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0cm9KuEeMLGI"
      },
      "source": [
        "대신 `tf.ragged.constant`를 사용하여 `tf.RaggedTensor`를 작성합니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.219190Z",
          "iopub.status.busy": "2021-09-22T20:44:53.218542Z",
          "iopub.status.idle": "2021-09-22T20:44:53.221194Z",
          "shell.execute_reply": "2021-09-22T20:44:53.220675Z"
        },
        "id": "XhF3QV3jiqTj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ccc6411d-950d-44aa-af5e-1cb658e70ba0"
      },
      "source": [
        "ragged_tensor = tf.ragged.constant(ragged_list)\n",
        "print(ragged_tensor)"
      ],
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sFgHduHVNoIE"
      },
      "source": [
        "`tf.RaggedTensor`의 형상에는 알 수 없는 길이의 일부 축이 포함됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.225820Z",
          "iopub.status.busy": "2021-09-22T20:44:53.225179Z",
          "iopub.status.idle": "2021-09-22T20:44:53.227368Z",
          "shell.execute_reply": "2021-09-22T20:44:53.227840Z"
        },
        "id": "Eo_3wJUWNgqB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "48e51a9b-d8f0-49ec-e3c7-5117e6d7be55"
      },
      "source": [
        "print(ragged_tensor.shape)"
      ],
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(4, None)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "V9njclVkkN7G"
      },
      "source": [
        "## 문자열 텐서(String tensors)\n",
        "\n",
        "`tf.string`은 `dtype`이며, 텐서에서 문자열(가변 길이의 바이트 배열)과 같은 데이터를 나타낼 수 있습니다.\n",
        "\n",
        "문자열은 원자성이므로 Python 문자열과 같은 방식으로 인덱싱할 수 없습니다. 문자열의 길이는 텐서의 축 중의 하나가 아닙니다. 문자열을 조작하는 함수에 대해서는 [`tf.strings`](https://www.tensorflow.org/api_docs/python/tf/strings)를 참조하세요."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5P_8spEGQ0wp"
      },
      "source": [
        "다음은 스칼라 문자열 텐서입니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.233278Z",
          "iopub.status.busy": "2021-09-22T20:44:53.232612Z",
          "iopub.status.idle": "2021-09-22T20:44:53.234830Z",
          "shell.execute_reply": "2021-09-22T20:44:53.235242Z"
        },
        "id": "sBosmM8MkIh4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f86903bb-d7f6-4d5a-d432-fde6a2af800e"
      },
      "source": [
        "# 텐서는 문자열이 될 수 있으며 여기에 스칼라 문자열이 있습니다.\n",
        "scalar_string_tensor = tf.constant(\"Gray wolf\")\n",
        "print(scalar_string_tensor)"
      ],
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor(b'Gray wolf', shape=(), dtype=string)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CMFBSl1FQ3vE"
      },
      "source": [
        "문자열의 벡터는 다음과 같습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IO-c3Tq3RC1L"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th>문자열 벡터, shape: <code>[3,]</code></th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/strings.png\" alt=\"문자열의 길이는 텐서의 축 중 하나가 아니다.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.240491Z",
          "iopub.status.busy": "2021-09-22T20:44:53.239863Z",
          "iopub.status.idle": "2021-09-22T20:44:53.242063Z",
          "shell.execute_reply": "2021-09-22T20:44:53.242493Z"
        },
        "id": "41Dv2kL9QrtO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b1dbdb7e-859a-4e66-ffcc-046b7e405008"
      },
      "source": [
        "# 길이가 다른 문자열 텐서가 세 개가 있다.\n",
        "tensor_of_strings = tf.constant([\"Gray wolf\",\n",
        "                                 \"Quick brown fox\",\n",
        "                                 \"Lazy dog\"])\n",
        "# 형상이 (3, )입니다. 문자열 길이가 포함되지 않았습니다.\n",
        "print(tensor_of_strings)"
      ],
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "76gQ9qrgSMzS"
      },
      "source": [
        "위의 출력에서 `b` 접두사는 `tf.string dtype`이 유니코드 문자열이 아니라 바이트 문자열임을 나타냅니다. TensorFlow에서 유니코드 텍스트를 처리하는 자세한 내용은 [유니코드 튜토리얼](https://www.tensorflow.org/tutorials/load_data/unicode)을 참조하세요."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ClSBPK-lZBQp"
      },
      "source": [
        "유니코드 문자를 전달하면 UTF-8로 인코딩됩니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.248159Z",
          "iopub.status.busy": "2021-09-22T20:44:53.247425Z",
          "iopub.status.idle": "2021-09-22T20:44:53.249866Z",
          "shell.execute_reply": "2021-09-22T20:44:53.250295Z"
        },
        "id": "GTgL53jxSMd9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c29c20e9-5e91-4625-9ee3-0020aaa0ed6d"
      },
      "source": [
        "tf.constant(\"🥳👍\")"
      ],
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(), dtype=string, numpy=b'\\xf0\\x9f\\xa5\\xb3\\xf0\\x9f\\x91\\x8d'>"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ir9cY42MMAei"
      },
      "source": [
        "문자열이 있는 일부 기본 함수는 `tf.strings`을 포함하여 `tf.strings.split`에서 찾을 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.255347Z",
          "iopub.status.busy": "2021-09-22T20:44:53.254667Z",
          "iopub.status.idle": "2021-09-22T20:44:53.264386Z",
          "shell.execute_reply": "2021-09-22T20:44:53.264836Z"
        },
        "id": "8k2K0VTFyj8e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cd82b307-5ac0-4fd6-8adb-8f93368a17e1"
      },
      "source": [
        "# 분할을 사용하여 문자열을 텐서 세트로 분할할 수 있습니다.\n",
        "print(tf.strings.split(scalar_string_tensor, sep=\" \"))"
      ],
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.270225Z",
          "iopub.status.busy": "2021-09-22T20:44:53.269513Z",
          "iopub.status.idle": "2021-09-22T20:44:53.276287Z",
          "shell.execute_reply": "2021-09-22T20:44:53.275739Z"
        },
        "id": "zgGAn1dfR-04",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4bb77e6d-4aae-4690-b5d0-8183806e8876"
      },
      "source": [
        "# 하지만 문자열로 된 텐서를 쪼개면 `비정형텐서(RaggedTensor)`로 변합니다.\n",
        "# 따라서 각 문자열은 서로 다른 수의 부분으로 분할될 수 있습니다.\n",
        "print(tf.strings.split(tensor_of_strings))"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HsAn1kPeO84m"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th>세 개의 분할된 문자열, shape: <code>[3, None]</code></th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/string-split.png\" alt=\"Splitting multiple strings returns a tf.RaggedTensor\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "st9OxrUxWSKY"
      },
      "source": [
        "`tf.string.to_number`:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.281468Z",
          "iopub.status.busy": "2021-09-22T20:44:53.279155Z",
          "iopub.status.idle": "2021-09-22T20:44:53.288254Z",
          "shell.execute_reply": "2021-09-22T20:44:53.288708Z"
        },
        "id": "3nRtx3X9WRfN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0cdec080-06c1-458f-b6be-45eaf65f2c5c"
      },
      "source": [
        "text = tf.constant(\"1 10 100\")\n",
        "print(tf.strings.to_number(tf.strings.split(text, \" \")))"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r2EZtBbJBns4"
      },
      "source": [
        "`tf.cast`를 사용하여 문자열 텐서를 숫자로 변환할 수는 없지만, 바이트로 변환한 다음 숫자로 변환할 수 있습니다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.294040Z",
          "iopub.status.busy": "2021-09-22T20:44:53.293350Z",
          "iopub.status.idle": "2021-09-22T20:44:53.300876Z",
          "shell.execute_reply": "2021-09-22T20:44:53.301275Z"
        },
        "id": "fo8BjmH7gyTj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f2ec0c44-62e6-4bc3-89c5-4f1e6b5bf801"
      },
      "source": [
        "byte_strings = tf.strings.bytes_split(tf.constant(\"Duck\"))\n",
        "byte_ints = tf.io.decode_raw(tf.constant(\"Duck\"), tf.uint8)\n",
        "print(\"Byte strings:\", byte_strings)\n",
        "print(\"Bytes:\", byte_ints)"
      ],
      "execution_count": 68,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Byte strings: tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)\n",
            "Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.306840Z",
          "iopub.status.busy": "2021-09-22T20:44:53.306105Z",
          "iopub.status.idle": "2021-09-22T20:44:53.313053Z",
          "shell.execute_reply": "2021-09-22T20:44:53.313495Z"
        },
        "id": "uSQnZ7d1jCSQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "be6b20f2-a289-44ef-ff9c-f1e0523df18d"
      },
      "source": [
        "# 또는 유니코드로 분할한 다음 디코딩합니다.\n",
        "unicode_bytes = tf.constant(\"アヒル 🦆\")\n",
        "unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, \"UTF-8\")\n",
        "unicode_values = tf.strings.unicode_decode(unicode_bytes, \"UTF-8\")\n",
        "\n",
        "print(\"\\nUnicode bytes:\", unicode_bytes)\n",
        "print(\"\\nUnicode chars:\", unicode_char_bytes)\n",
        "print(\"\\nUnicode values:\", unicode_values)"
      ],
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Unicode bytes: tf.Tensor(b'\\xe3\\x82\\xa2\\xe3\\x83\\x92\\xe3\\x83\\xab \\xf0\\x9f\\xa6\\x86', shape=(), dtype=string)\n",
            "\n",
            "Unicode chars: tf.Tensor([b'\\xe3\\x82\\xa2' b'\\xe3\\x83\\x92' b'\\xe3\\x83\\xab' b' ' b'\\xf0\\x9f\\xa6\\x86'], shape=(5,), dtype=string)\n",
            "\n",
            "Unicode values: tf.Tensor([ 12450  12498  12523     32 129414], shape=(5,), dtype=int32)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fE7nKJ2YW3aY"
      },
      "source": [
        "`tf.string` dtype은 TensorFlow의 모든 원시 바이트 데이터에 사용됩니다. `tf.io` 모듈에는 이미지 디코딩 및 csv 구문 분석을 포함하여 데이터를 바이트로 변환하거나 바이트에서 변환하는 함수가 포함되어 있습니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ua8BnAzxkRKV"
      },
      "source": [
        "## 희소 텐서(Sparse tensors)\n",
        "\n",
        "때로는 매우 넓은 임베드 공간과 같이 데이터가 희소합니다. TensorFlow는 `tf.sparse.SparseTensor` 및 관련 연산을 지원하여 희소 데이터를 효율적으로 저장합니다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mS5zgqgUTPRb"
      },
      "source": [
        "<table>\n",
        "<tr>\n",
        "  <th>A `tf.SparseTensor`, shape: <code>[3, 4]</code></th>\n",
        "</tr>\n",
        "<tr>\n",
        "  <td>\n",
        "<img src=\"img/sparse.png\" alt=\"셀 중 두 개에만 값이 있는 3x4 그리드.\">\n",
        "  </td>\n",
        "</tr>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2021-09-22T20:44:53.319583Z",
          "iopub.status.busy": "2021-09-22T20:44:53.318890Z",
          "iopub.status.idle": "2021-09-22T20:44:53.323982Z",
          "shell.execute_reply": "2021-09-22T20:44:53.323486Z"
        },
        "id": "B9nbO1E2kSUN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4bd9775d-cd5d-4ecd-f369-8245ad692bd4"
      },
      "source": [
        "# 희소 텐서는 메모리 효율적 방식으로 인덱스별로 값을 저장한다.\n",
        "sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],\n",
        "                                       values=[1, 2],\n",
        "                                       dense_shape=[3, 4])\n",
        "print(sparse_tensor, \"\\n\")\n",
        "\n",
        "# 희소 텐서를 고밀도(dense) 텐서로 변환할 수 있습니다.\n",
        "print(tf.sparse.to_dense(sparse_tensor))"
      ],
      "execution_count": 70,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "SparseTensor(indices=tf.Tensor(\n",
            "[[0 0]\n",
            " [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) \n",
            "\n",
            "tf.Tensor(\n",
            "[[1 0 0 0]\n",
            " [0 0 2 0]\n",
            " [0 0 0 0]], shape=(3, 4), dtype=int32)\n"
          ]
        }
      ]
    }
  ]
}
