defmodule Mlixir.StochasticGradientDescent do
  @moduledoc """
  Model representing stochastic gradient descent.
  """

  @behaviour Mlixir.Model

  import Nx.Defn

  @learning_rate 0.01

  @epochs 1_000

  @doc """
  Train the SGD model.

  Uses mean squared error by default.
  """
  @impl true
  def fit(%Nx.Tensor{shape: {_, n_coef}} = x, y) do
    coefficients = Nx.broadcast(0, {n_coef})
    Enum.reduce(0..@epochs, coefficients, fn _, acc -> gradient_step(acc, x, y) end)
  end

  defnp gradient_step(coefficients, x, y) do
    error = grad(coefficients, fn coeff -> loss(coeff, x, y) end)
    coefficients - (error * @learning_rate)
  end

  defnp loss(coefficients, x, y) do
    y_pred = Nx.dot(x, coefficients)
    mean_squared_error(y_pred, y)
  end

  defnp mean_squared_error(y_pred, y) do
    {n_samples} = Nx.shape(y_pred)

    y - y_pred
    |> Nx.power(2)
    |> Nx.sum()
    |> Nx.divide(n_samples)
  end

  @doc """
  Predict the dependent variable value for given input.
  """
  @impl true
  defn predict(model, x) do
    Nx.dot(model, x)
  end
end