defmodule Mlixir.StochasticGradientDescent do
  @moduledoc """
  Model representing stochastic gradient descent.
  """

  @behaviour Mlixir.Model

  alias Mlixir.Shared

  import Nx.Defn

  @doc """
  Train the SGD model.
  """
  @impl true
  defn fit(x, y) do
  end

  @doc """
  Predict the dependent variable value for given input.
  """
  @impl true
  defn predict(model, x) do
  end
end
