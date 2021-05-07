defmodule Mlixir.KNN do
  @moduledoc """
  Model representing a k-nearest neighbours regressor/classifier.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  """
  @impl true
  defn fit(x, y) do
    {x, y}
  end

  @doc """
  """
  @impl true
  defn predict({x_model, _}, x) do
    x
    |> Nx.subtract(x_model)
    |> batch_euclidean_norm()
    |> Nx.argsort()
  end

  defnp batch_euclidean_norm(t) do
    t
    |> Nx.power(2)
    |> Nx.sum(axes: [1])
    |> Nx.power(0.5)
  end
end
