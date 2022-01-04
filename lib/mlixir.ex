defmodule Mlixir do
  @moduledoc """
  Mlixir - Machine Learning in Elixir.
  """

  import Nx.Defn

  @doc """
  Left pads a tensor with the given scalar.
  """
  defn left_pad(t, value) do
    config = transform(Nx.rank(t), & List.duplicate({0, 0, 0}, &1 - 1) ++ [{1, 0, 0}])
    Nx.pad(t, value, config)
  end

  defn relu(t) do
    Nx.select(t > 0, t, 0)
  end

  defn reverse_relu(t) do
    Nx.select(t > 0, 0, t)
  end

  defn scatter(t1, indices, t2) do
    {n, _} = shape = Nx.shape(t1)
    ones = Nx.broadcast(1, {n})

    Nx.broadcast(0, shape)
    |> Nx.indexed_add(indices, ones)
    |> Nx.select(t2, t1)
  end

  defn fill_diagonal(t, v) do
    t
    |> Nx.eye()
    |> Nx.select(v, t)
  end
end
