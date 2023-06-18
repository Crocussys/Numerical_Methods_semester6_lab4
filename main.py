import numpy as np
import matplotlib.pyplot as plt
import math


def task1(u, t, w, **kwargs):
    u1, u2 = u[0], u[1]
    a = 2.5 + w / 40
    return np.array([-u1 * u2 + math.sin(t) / t, -(u2 ** 2) + a * t / (1 + t ** 2)])


def task2(u, a, k, **kwargs):
    u1, u2 = u[0], u[1]
    return np.array([u2 - u1 * (a * u1 + k * u2), math.exp(u1) - u1 * (u1 + a * u2)])


def task3(u, a, k, **kwargs):
    u1, u2, u3 = u[0], u[1], u[2]
    return np.array([u2 * u3 * (k - a) / a, u1 * u3 * (a + k) / k, u1 * u2 * (a - k) / a])


def f_for_newton(task_func, x, y, a0, a1, b0, *args, **kwargs):
    return x - a1 * y[0] - a0 * y[1] - b0 * task_func(x, *args, **kwargs)


def jacobi_for_task1(x, t, tau_k, *args, **kwargs):
    u1, u2 = x[0], x[1]
    det = 1 + 3 * u2 * tau_k + 2 * u2 ** 2 * tau_k ** 2
    return np.array([[1 + 2 * u2 * tau_k, -u1 * tau_k], [0, 1 + u2 * tau_k]]) / det


def jacobi_for_task2(x, t, tau_k, a, k,  *args, **kwargs):
    u1, u2 = x[0], x[1]
    det = 1 + tau_k * (3 * u1 * a + k * u2) + tau_k ** 2 * ((u1 * k - 1) * (math.exp(u1) - 2 * u1) + a *
                                                            (u2 + 2 * u1 * a))
    return np.array([[1 + u1 * tau_k * a, tau_k - u1 * tau_k * k],
                     [tau_k * math.exp(u1) - 2 * u1 * tau_k - a * u2 * tau_k,
                      1 + 2 * u1 * tau_k * a + u2 * tau_k * k]]) / det


def jacobi_for_task3(x, t, tau_k, a, k,  *args, **kwargs):
    u1, u2, u3 = x[0], x[1], x[2]
    b1 = (a + k) / k
    b2 = (a - k) / a
    det = 1 + 2 * b2 ** 2 * b1 * u1 * u2 * u3 * tau_k ** 3 + (b2 * u2 * tau_k) ** 2 - b2 * b1 * (u1 ** 2 - u3 ** 2) *\
          tau_k ** 2
    return np.array([[1 - b1 * b2 * u1 ** 2 * tau_k ** 2, -b2 * u3 * tau_k - b2 ** 2 * u1 * u2 * tau_k ** 2,
                      b1 * -b2 * u1 * u3 * tau_k ** 2 - b2 * u2 * tau_k],
                     [b1 * u3 * tau_k + b1 * b2 * u1 * u2 * tau_k ** 2, 1 + (b2 * u2 * tau_k) ** 2,
                      b1 * u1 * tau_k - b2 * b1 * u2 * u3 * tau_k ** 2],
                     [b1 * b2 * u1 * u3 * tau_k ** 2 + b2 * u2 * tau_k,
                      b2 * u1 * tau_k - b2 ** 2 * u2 * u3 * tau_k ** 2, 1 + b2 * b1 * u3 ** 2 * tau_k ** 2]]) / det


def chart(title, xs, ys):
    fig, ax = plt.subplots()
    for i in range(len(ys)):
        for j in range(len(ys[i])):
            ax.plot(xs, ys[i][j], label=f"Task {i + 1} u{j + 1}")
    ax.grid(which='major', color='gray')
    ax.grid(which='minor', color='gray', linestyle=':')
    ax.legend()
    plt.xlabel("t", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title(title)
    plt.savefig(f"Charts\\{title}.svg".replace("\n", " "))
    plt.show()


def euler_explicit(func, u0, interval, eps, tau_interval, *args, **kwargs):
    t = interval[0]
    t_end = interval[1]
    tau_max = tau_interval[1]
    y = u0
    i = 1
    xs, ys = list(), [[] for _ in range(len(y))]
    while t < t_end:
        kwargs.update({"t": t})
        f = func(y, *args, **kwargs)
        tau = np.amin(eps / (abs(f) + eps / tau_max))
        y += tau * f
        t += tau
        print(f"{i}: {y} {t}")
        xs.append(t)
        for j in range(len(y)):
            ys[j].append(y[j])
        i += 1
    return xs, ys


def is_end(x_prev, x, fx, err1, err2):
    delta1, delta2 = max(abs(fx)), 0
    for i in range(len(x)):
        if abs(x[i]) < 1:
            val = abs(x[i] - x_prev[i])
        else:
            val = abs(1 - x_prev[i] / x[i])
        if val > delta2:
            delta2 = val
    if delta1 > err1:
        return False
    if delta2 > err2:
        return False
    return True


def newton(func, func_jacobi, x, max_i=1000, err1=1e-9, err2=1e-9, *args, **kwargs):
    x_prev = x.copy()
    i, fx = 1, func(x=x_prev, *args, **kwargs)
    x_next = x_prev - func_jacobi(x=x_prev, *args, **kwargs) @ fx
    while not is_end(x_prev, x_next, fx, err1, err2) and i <= max_i:
        x_prev = x_next
        fx = func(x=x_prev, *args, **kwargs)
        x_next = x_prev - func_jacobi(x=x_prev, *args, **kwargs) @ fx
        i += 1
    return x_next


def euler_implicit(newton_func, jacobi_func, u0, interval, eps, tau_interval, print_ever=1, *args, **kwargs):
    t = interval[0]
    t_end = interval[1]
    y = [u0] * 3
    tau = [tau_interval[0], tau_interval[0], None]
    tau_max = tau_interval[1]
    i = 1
    xs, ys = list(), [[] for _ in range(len(u0))]
    while t < t_end:
        t += tau[1]
        y[2] = newton(newton_func, jacobi_func, y[2], err1=eps, err2=eps, y=y, t=t, a0=1, a1=0, b0=tau[1], tau_k=tau[1],
                      *args, **kwargs)
        eps_k = -tau[1] * (y[2] - y[1] - tau[1] * (y[1] - y[0]) / tau[0]) / (tau[1] + tau[0])
        eps_k = np.amin(eps_k)
        if abs(eps_k) > eps:
            tau[2] = tau[1] / 2
        elif eps / 4 < abs(eps_k) <= eps:
            tau[2] = tau[1]
        else:
            tau[2] = 2 * tau[1]
        if tau[2] > tau_max:
            tau[2] = tau_max
        if i % print_ever == 0:
            print(f"{i}: {y[2]} {t}")
        xs.append(t)
        for j in range(len(u0)):
            ys[j].append(y[2][j])
        y[0], y[1], tau[0], tau[1] = y[1], y[2], tau[1], tau[2]
        i += 1
    return xs, ys


def shikhman(newton_func, jacobi_func, u0, interval, eps, tau_interval, print_ever=1, *args, **kwargs):
    t = interval[0]
    t_end = interval[1]
    y = [u0] * 3
    tau = [tau_interval[0], tau_interval[0], None]
    tau_max = tau_interval[1]
    i = 1
    xs, ys = list(), [[] for _ in range(len(u0))]
    while t < t_end:
        t += tau[1]
        if i == 1 or i == 2:
            y[2] = newton(newton_func, jacobi_func, y[2], err1=eps, err2=eps, y=y, t=t, a0=1, a1=0, b0=tau[1],
                          tau_k=tau[1], *args, **kwargs)
        else:
            a0 = (tau[1] + tau[0]) ** 2 / tau[0] / (2 * tau[1] + tau[0])
            a1 = - tau[1] ** 2 / tau[0] / (2 * tau[1] + tau[0])
            b0 = tau[1] * (tau[1] + tau[0]) / (2 * tau[1] + tau[0])
            y[2] = newton(newton_func, jacobi_func, y[2], err1=eps, err2=eps, y=y, t=t, a0=a0, a1=a1, b0=b0,
                          tau_k=tau[1], *args, **kwargs)
        eps_k = -tau[1] * (y[2] - y[1] - tau[1] * (y[1] - y[0]) / tau[0]) / (tau[1] + tau[0])
        eps_k = np.amin(eps_k)
        if abs(eps_k) > eps:
            tau[2] = tau[1] / 2
        elif eps / 4 < abs(eps_k) <= eps:
            tau[2] = tau[1]
        else:
            tau[2] = 2 * tau[1]
        if tau[2] > tau_max:
            tau[2] = tau_max
        if i % print_ever == 0:
            print(f"{i}: {y[2]} {t}")
        xs.append(t)
        for j in range(len(u0)):
            ys[j].append(y[2][j])
        y[0], y[1], tau[0], tau[1] = y[1], y[2], tau[1], tau[2]
        i += 1
    return xs, ys


def combine_answers(new_xs, new_ys, xs=None, ys=None):
    if ys is None:
        return new_xs, [new_ys]
    size_xs = len(xs)
    size_new_xs = len(new_xs)
    if size_new_xs < size_xs:
        for _ in range(size_xs - size_new_xs):
            new_xs.append(None)
    elif size_new_xs > size_xs:
        for _ in range(size_new_xs - size_xs):
            xs.append(None)
            for i in range(len(ys)):
                for j in range(len(ys[0])):
                    ys[i][j].append(None)
    ys.append(new_ys)
    return xs, ys


if __name__ == '__main__':
    xss1, yss1 = euler_explicit(task1, np.array([0.0, -0.412]), (0.001, 1), 1e-3, (0.001, 0.1), w=25)
    xss, yss = combine_answers(xss1, yss1)
    xss2, yss2 = euler_explicit(task1, np.array([0.0, -0.412]), (0.001, 1), 1e-3, (0.001, 0.1), w=48)
    xss, yss = combine_answers(xss2, yss2, xss, yss)
    chart("Явный метод Эйлера\nЗадача 1", xss, yss)
    xss, yss = euler_explicit(task2, np.array([1.0, 0.0]), (0.001, 1), 1e-3, (0.001, 0.1), a=2, k=0.25)
    chart("Явный метод Эйлера\nЗадача 2", xss, [yss])
    xss, yss = euler_explicit(task3, np.array([1.0, 1.0, 1.0]), (0.001, 1), 1e-3, (0.001, 0.1), a=2, k=0.25)
    chart("Явный метод Эйлера\nЗадача 3", xss, [yss])
    xss, yss = euler_implicit(f_for_newton, jacobi_for_task1, np.array([0.0, -0.412]), (0, 1), 1e-3, (0.001, 0.1),
                              print_ever=1, task_func=task1, w=25)
    chart("Неявный метод Эйлера\nЗадача 1", xss, [yss])
    xss, yss = euler_implicit(f_for_newton, jacobi_for_task2, np.array([1.0, 0.0]), (0, 1), 1e-3, (0.001, 0.1),
                              print_ever=1, task_func=task2, a=2, k=0.25)
    chart("Неявный метод Эйлера\nЗадача 2", xss, [yss])
    xss, yss = euler_implicit(f_for_newton, jacobi_for_task3, np.array([1.0, 1.0, 1.0]), (0.001, 1), 1e-3, (0.001, 0.1),
                              print_ever=1, task_func=task3, a=2, k=0.25)
    chart("Неявный метод Эйлера\nЗадача 3", xss, [yss])
    xss, yss = shikhman(f_for_newton, jacobi_for_task1, np.array([0.0, -0.412]), (0, 1), 1e-3, (0.001, 0.1),
                        print_ever=1, task_func=task1, w=25)
    chart("Метод Шихмана\nЗадача 1", xss, [yss])
    xss, yss = shikhman(f_for_newton, jacobi_for_task2, np.array([1.0, 0.0]), (0, 1), 1e-3, (0.001, 0.1), print_ever=1,
                        task_func=task2, a=2, k=0.25)
    chart("Метод Шихмана\nЗадача 2", xss, [yss])
    xss, yss = shikhman(f_for_newton, jacobi_for_task3, np.array([1.0, 1.0, 1.0]), (0.001, 1), 1e-3, (0.001, 0.1),
                        print_ever=1, task_func=task3, a=2, k=0.25)
    chart("Метод Шихмана\nЗадача 3", xss, [yss])
