from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.
    descent_config : dict - Конфигурация градиентного спуска.
    tolerance : float, optional - Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional - Критерий остановки по количеству итераций. По умолчанию равен 300.
    descent : BaseDescent - Экземпляр класса, реализующего градиентный спуск.
    tolerance : float - Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int - Критерий остановки по количеству итераций.
    loss_history : List[float] - История значений функции потерь на каждой итерации.
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        self : LinearRegression - Возвращает экземпляр класса с обученными весами.
        """
        initial_loss = self.descent.calc_loss(x, y)
        self.loss_history.append(initial_loss)

        for iteration in range(self.max_iter):
            delta_w = self.descent.step(x, y)

            if np.any(np.isnan(self.descent.w)):
                print("NaN")
                break

            current_loss = self.descent.calc_loss(x, y)
            self.loss_history.append(current_loss)

            if np.linalg.norm(delta_w) ** 2 < self.tolerance:
                print(f"Разность весов меньше порога ({self.tolerance}).")
                break

        return self

        raise NotImplementedError('Функция fit класса LinearRegression не реализована')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.
        x : np.ndarray - Массив признаков.
        prediction : np.ndarray - Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        loss : float - Значение функции потерь.
        """
        self.grad = (2/len(y))*x.T@(np.dot(x,self.w) - y)
        return self.descent.calc_loss(x, y)
