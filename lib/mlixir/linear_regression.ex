defmodule Mlixir.LinearRegression do
  @moduledoc """
  Model representing multivariate linear regression.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  Train the regression model using the Ordinary Least Square method.
  """
  @impl true
  defn fit(x, y) do
    x_padded = Mlixir.left_pad(x, 1)
    x_transposed = Nx.transpose(x_padded)
    first = Nx.LinAlg.invert(Nx.dot(x_transposed, x_padded))
    second = Nx.dot(x_transposed, y)
    Nx.dot(first, second)
  end

  @doc """
  Predict the dependent variable value for given input.
  """
  @impl true
  defn predict(model, x) do
    Nx.dot(model, Mlixir.left_pad(x, 1))
  end
end
