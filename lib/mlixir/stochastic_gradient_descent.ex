defmodule Mlixir.StochasticGradientDescent do
  @moduledoc """
  Model representing stochastic gradient descent.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  Train the SGD model.

  Uses mean squared error by default.
  """
  @impl true
  defn fit(x, y, epochs \\ 1_000, learning_rate \\ 0.1) do
    x_padded = Mlixir.left_pad(x, 1)
    {_, n_coef} = Nx.shape(x_padded)
    coefficients = Nx.broadcast(0, {n_coef})
    coefficients_expr = transform(coefficients, &Nx.Defn.Expr.tensor/1)

    transform(
      {x_padded, y, coefficients_expr, epochs, learning_rate},
      fn {x_padded, y, coefficients_expr, epochs, learning_rate} ->
        Enum.reduce(
          0..epochs,
          coefficients_expr,
          fn _, acc -> gradient_step(acc, x_padded, y, learning_rate) end
        )
      end
    )
  end

  defnp gradient_step(coefficients, x, y, learning_rate) do
    error = grad(coefficients, fn coeff -> loss(coeff, x, y) end)
    coefficients - (error * learning_rate)
  end

  defnp loss(coefficients, x, y) do
    y_pred = Nx.dot(x, coefficients)
    mean_squared_error(y_pred, y)
  end

  defnp mean_squared_error(y_pred, y) do
    {n_samples} = Nx.shape(y_pred)

    y
    |> Nx.subtract(y_pred)
    |> Nx.power(2)
    |> Nx.sum()
    |> Nx.divide(n_samples)
  end

  @doc """
  Predict the dependent variable value for given input.
  """
  @impl true
  defn predict(model, x) do
    Nx.dot(model, Mlixir.left_pad(x, 1))
  end
end
