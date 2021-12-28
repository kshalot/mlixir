defmodule Mlixir.AffinityPropagation do
  @moduledoc """
  Model representing affinity propagation clustering.
  """

  import Nx.Defn

  @iterations 10

  @neg_inf -100000000

  @doc """
  Cluster the dataset using affinity propagation.
  """
  defn fit(data) do
    {initial_a, initial_r, s} = initialize_matrices(data)
    {final_a, final_r, _, _} =
      while {a = initial_a, r = initial_r, s = s, i = @iterations}, Nx.greater(i, 0) do
        {new_a, new_r} = propagate_matrices(a, r, s)
        {new_a, new_r, s, i - 1}
      end
  end

  defnp propagate_matrices(a, r, s) do
    {propagate_availabilities(a, r), propagate_responsibilities(a, r, s)}
  end

  defnp propagate_responsibilities(a, r, s) do
    temp = a + s

    max_local_indices = temp
    |> Nx.argmax(axis: 1)
    |> Nx.new_axis(1)
    first_maxes = temp
    |> Nx.take_along_axis(max_local_indices, axis: 1)

    {n, _} = shape = Nx.shape(a)
    infinities = Nx.broadcast(@neg_inf, {n})
    max_indices = Nx.stack([Nx.iota({n}), max_local_indices])
    |> Nx.transpose()
    second_maxes = temp
    |> Nx.indexed_add(max_indices, infinities)
    |> Nx.reduce_max(axes: [1])

    max_matrix = Nx.broadcast(0.0, shape) + first_maxes
    ones = Nx.broadcast(1, {n})
    pred = Nx.broadcast(0, shape)
    |> Nx.indexed_add(max_indices, ones)

    max_matrix = pred
    |> Nx.select(second_maxes, max_matrix)

    new_val = s - max_matrix
    r + new_val
  end

  defnp propagate_availabilities(a, _r) do
    # TODO: Implement (k, k) indices case
    a
    # |> relu()
    # |> Nx.sum(axes: [1]) # FIXME: current indices have to be skipped
    # |> Nx.add(Nx.take_diagonal(r))
    # |> reverse_relu()
  end

  defnp initialize_matrices(data) do
    {n, _} = Nx.shape(data)
    availability_matrix = Nx.broadcast(0.0, {n, n})
    responsibility_matrix = Nx.broadcast(0.0, {n, n})
    similarity_matrix = initialize_similarities(data)

    {availability_matrix, responsibility_matrix, similarity_matrix}
  end

  defnp initialize_similarities(data) do
    {n, dims} = Nx.shape(data)
    t1 = Nx.reshape(data, {1, n, dims})
    t2 = Nx.reshape(data, {n, 1, dims})

    t1 - t2
    |> Nx.power(2)
    |> Nx.sum(axes: [2])
    |> Nx.power(0.5)
  end

  defnp relu(t) do
    Nx.select(t > 0, t, 0)
  end

  defnp reverse_relu(t) do
    Nx.select(t > 0, 0, t)
  end
end
