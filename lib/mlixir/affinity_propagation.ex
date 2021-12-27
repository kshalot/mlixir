defmodule Mlixir.AffinityPropagation do
  @moduledoc """
  Model representing affinity propagation clustering.
  """

  import Nx.Defn

  @iterations 100

  @doc """
  Cluster the dataset using affinity propagation.
  """
  defn fit(data) do
    {initial_a, initial_r} = initialize_matrices(Nx.shape(data))
    {final_a, final_r, _} =
      while {a = initial_a, r = initial_r, i = @iterations}, Nx.greater(i, 1) do
        {new_a, new_r} = propagate_matrices(a, r)
        {new_a, new_r, i - 1}
      end
  end

  defnp initialize_matrices({n, _}) do
    availability_matrix = Nx.broadcast(0, {n, n})
    responsibility_matrix = Nx.broadcast(0, {n, n})

    {availability_matrix, responsibility_matrix}
  end

  defnp propagate_matrices(a, r) do
    {propagate_availabilities(a), propagate_responsibilities(r)}
  end

  defnp propagate_responsibilities(r) do
    r
  end

  defnp propagate_availabilities(a) do
    a
  end
end
