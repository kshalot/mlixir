defmodule Mlixir.SVM do
  @moduledoc """
  Model representing a Support-Vector Machine.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  Train the SVM using stochastic gradient descent.
  """
  @impl true
  defn fit(x, y, epochs \\ 1_000, learning_rate \\ 0.1) do
    Mlixir.SGD.fit(x, y, epochs, learning_rate)
  end

  @doc """
  Predict the class (binary, 1 or -1) of the given input.
  """
  @impl true
  defn predict(model, x) do
    model
    |> Mlixir.SGD.predict(x)
    |> Nx.greater(0)
    |> Nx.multiply(2)
    |> Nx.subtract(1)
  end
end
