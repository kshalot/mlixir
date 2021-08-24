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
    x_transposed = Nx.transpose(x)
    first = Nx.LinAlg.invert(Nx.dot(x_transposed, x))
    second = Nx.dot(x_transposed, y)
    Nx.dot(first, second)
  end

  @doc """
  Predict the dependent variable value for given input.
  """
  @impl true
  defn predict(model, x) do
    #Nx.dot(Mlixir.left_pad(x, 1), model)
    Nx.dot(x, model)
  end
end
