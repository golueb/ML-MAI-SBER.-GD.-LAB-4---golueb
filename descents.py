from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.
    lambda_ : float, optional - Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional - Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional - Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional - Текущая итерация. По умолчанию 0.
    __call__() - Вычисляет скорость обучения на текущей итерации.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Вычисляет скорость обучения по формуле lambda * (s0 / (s0 + t))^p.
        float - Скорость обучения на текущем шаге.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.
    MSE : auto - Среднеквадратическая ошибка.
    MAE : auto - Средняя абсолютная ошибка.
    LogCosh : auto - Логарифм гиперболического косинуса от ошибки.
    Huber : auto - Функция потерь Хьюбера.
    """
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    def cals_loss(self, y,ans):
        self.error = ((y - ans)**2).mean()
        return self.error
    def predict(self, x):
        self.pred = np.dot(self.w,x)
        return self.pred
    
    """
    Базовый класс для всех методов градиентного спуска.
    dimension : int - Размерность пространства признаков.
    lambda_ : float, optional - Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional - Функция потерь, которая будет оптимизироваться. По умолчанию MSE.
    w : np.ndarray - Вектор весов модели.
    lr : LearningRate - Скорость обучения.
    loss_function : LossFunction - Функция потерь.
    
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray - Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray - Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray - Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float - Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray - Вычисление прогнозов на основе признаков x.
    """


    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация базового класса для градиентного спуска.
        dimension : int - Размерность пространства признаков.
        lambda_ : float - Параметр скорости обучения.
        loss_function : LossFunction - Функция потерь, которая будет оптимизирована.
        w : np.ndarray - Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate - Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction - Выбранная функция потерь.
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        np.ndarray - Разность между текущими и обновленными весами.
        """

        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.
        gradient : np.ndarray - Градиент функции потерь по весам.
        np.ndarray - Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для вычисления градиента функции потерь по весам. Должен быть переопределен в подклассах.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        np.ndarray - Градиент функции потерь по весам. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        losses = 1 / (len(y)) * np.sum((y - y_pred) ** 2)
        return  losses
        """
        Вычисление значения функции потерь с использованием текущих весов.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        losses - Значение функции потерь.
        """
        raise NotImplementedError('BaseDescent calc_loss function not implemented')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.
        x : np.ndarray - Массив признаков.
        np.ndarray - Прогнозируемые значения.
        """
        return np.dot(x,self.w)
        raise NotImplementedError('BaseDescent predict function not implemented')


class VanillaGradientDescent(BaseDescent):
    """
    Класс полного градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray - Обновление весов с учетом градиента.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray - Вычисление градиента функции потерь по весам.
    """
    def __init__(self,dimension: int,  lambda_: float = 1e-3,loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension,lambda_, loss_function)
        self.k = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:

        s0 = 1
        p = 0.5
        lr = self.lr.lambda_ * (s0 / (s0 + self.k)) ** p
        self.deltaw = - gradient*lr
        self.w += self.deltaw
        self.k += 1
        return self.deltaw
        """
        Обновление весов на основе градиента.
        gradient : np.ndarray - Градиент функции потерь по весам.
        np.ndarray - Разность весов (w_{k + 1} - w_k).
        """
        raise NotImplementedError('VanillaGradientDescent update_weights function not implemented')

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.predict(x)
        return (2 / len(y)) * x.T @ (y_pred - y)
        """
        Вычисление градиента функции потерь по весам.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        np.ndarray - Градиент функции потерь по весам.
        """
        raise NotImplementedError('VanillaGradientDescent calc_gradient function not implemented')


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска.
    batch_size : int, optional - Размер мини-пакета. По умолчанию 50.
    batch_size : int - Размер мини-пакета.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray - Вычисление градиента функции потерь по мини-пакетам.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):

        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        self.li = np.random.randint(0, x.shape[0], size=self.batch_size)
        self.xx = x[self.li]
        self.yy = y[self.li]
        self.loss_grad = (2/len(self.li)) * self.xx.T @ ( (self.xx@self.w)-self.yy)
        return self.loss_grad

        """
        Вычисление градиента функции потерь по мини-пакетам.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        np.ndarray - Градиент функции потерь по весам, вычисленный по мини-пакету.
        """
        raise NotImplementedError('StochasticDescent calc_gradient function not implemented')


class MomentumDescent(VanillaGradientDescent):
    """
    Класс градиентного спуска с моментом.
    dimension : int - Размерность пространства признаков.
    lambda_ : float - Параметр скорости обучения.
    loss_function : LossFunction - Оптимизируемая функция потерь.
    alpha : float - Коэффициент момента.
    h : np.ndarray - Вектор момента для весов.
    update_weights(gradient: np.ndarray) -> np.ndarray - Обновление весов с использованием момента.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса градиентного спуска с моментом.
        dimension : int - Размерность пространства признаков.
        lambda_ : float - Параметр скорости обучения.
        loss_function : LossFunction - Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        s0 = 1
        p = 0.5
        lr = self.lr.lambda_ * (s0 / (s0 + self.k)) ** p
        self.h = self.alpha*self.h - lr*gradient
        self.deltaw = - self.h
        self.w += self.deltaw
        self.k += 1
        return self.deltaw
        """
        Обновление весов с использованием момента.
        gradient : np.ndarray - Градиент функции потерь.
        np.ndarray - Разность весов (w_{k + 1} - w_k).
        """
        raise NotImplementedError('MomentumDescent update_weights function not implemented')


class Adam(VanillaGradientDescent):
    """
    Класс градиентного спуска с адаптивной оценкой моментов (Adam).
    dimension : int - Размерность пространства признаков.
    lambda_ : float - Параметр скорости обучения.
    loss_function : LossFunction - Оптимизируемая функция потерь.
    eps : float - Малая добавка для предотвращения деления на ноль.
    m : np.ndarray - Векторы первого момента.
    v : np.ndarray - Векторы второго момента.
    beta_1 : float - Коэффициент распада для первого момента.
    beta_2 : float - Коэффициент распада для второго момента.
    update_weights(gradient: np.ndarray) -> np.ndarray - Обновление весов с использованием адаптивной оценки моментов.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса Adam.
        dimension : int - Размерность пространства признаков.
        lambda_ : float - Параметр скорости обучения.
        loss_function : LossFunction - Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов.
        gradient : np.ndarray - Градиент функции потерь.
        np.ndarray - Разность весов (w_{k + 1} - w_k).
        """
        self.iteration += 1
        self.m = self.m*self.beta_1 + (1 - self.beta_1) * gradient
        self.v = self.v*self.beta_2 + (1 - self.beta_2) * (gradient**2)
        self.m1 = self.m/(1 - self.beta_1**self.iteration)
        self.v1 = self.v/(1 - self.beta_2**self.iteration)
        s0 = 1
        p = 0.5
        lr = self.lr.lambda_ * (s0 / (s0 + self.k)) ** p
        self.deltaw = - lr/(np.sqrt(self.v1)+self.eps)

        return self.deltaw
        raise NotImplementedError('Adagrad update_weights function not implemented')


class BaseDescentReg(BaseDescent):
    """
    Базовый класс для градиентного спуска с регуляризацией.
    *args : tuple - Аргументы, передаваемые в базовый класс.
    mu : float, optional - Коэффициент регуляризации. По умолчанию равен 0.
    **kwargs : dict - Ключевые аргументы, передаваемые в базовый класс.
    mu : float - Коэффициент регуляризации.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray - Вычисление градиента функции потерь с учетом L2 регуляризации по весам.
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        Инициализация базового класса для градиентного спуска с регуляризацией.
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.
        x : np.ndarray - Массив признаков.
        y : np.ndarray - Массив целевых переменных.
        np.ndarray - Градиент функции потерь с учетом L2 регуляризации по весам.
        """
        l2_gradient: np.ndarray = np.zeros_like(x.shape[1])  

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    """


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1}
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
