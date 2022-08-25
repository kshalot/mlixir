defmodule Mlixir.AffinityPropagation do
  @moduledoc """
  Model representing affinity propagation clustering.
  """

  import Nx.Defn

  @iterations 200

  @inf 100_000_000

  @neg_inf -@inf

  @self_preference 0

  @damping_factor 0.5

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

    final_a + final_r
  end

  defnp propagate_matrices(a, r, s) do
    new_r = propagate_responsibilities(a, r, s)
    new_a = propagate_availabilities(a, new_r)

    {new_a, new_r}
  end

  defnp propagate_responsibilities(a, r, s) do
    temp = a + s

    max_local_indices = Nx.argmax(temp, axis: 1)
    first_maxes = temp
    |> Nx.take_along_axis(Nx.new_axis(max_local_indices, 1), axis: 1)

    {n, _} = shape = Nx.shape(a)
    infinities = Nx.broadcast(@neg_inf, {n})
    max_indices = Nx.stack([Nx.iota({n}), max_local_indices])
    |> Nx.transpose()
    second_maxes = temp
    |> Nx.indexed_add(max_indices, infinities)
    |> Nx.reduce_max(axes: [1])

    max_matrix = Nx.broadcast(0.0, shape) + first_maxes
    max_matrix = Mlixir.scatter(max_matrix, max_indices, second_maxes)

    res = s - max_matrix
    (r * @damping_factor) + ((1 - @damping_factor) * res)
  end

  defnp propagate_availabilities(a, r) do
    temp = r
    |> Mlixir.relu()
    |> Mlixir.fill_diagonal(0)
    |> Nx.sum(axes: [0])

    res = temp
    |> Nx.add(Nx.take_diagonal(r))
    |> Nx.broadcast(Nx.shape(r))
    |> Nx.subtract(Nx.clip(r, 0, @inf))
    |> Mlixir.reverse_relu()
    |> Mlixir.fill_diagonal(temp)

    (a * @damping_factor) - ((1 - @damping_factor) * res)
  end

  defnp initialize_matrices(data) do
    {n, _} = Nx.shape(data)
    availability_matrix = Nx.broadcast(0.0, {n, n})
    responsibility_matrix = Nx.broadcast(0.0, {n, n})
    similarity_matrix = initialize_similarities(data)

    {availability_matrix, responsibility_matrix, similarity_matrix}
  end

  defn initialize_similarities(data) do
    {n, dims} = Nx.shape(data)
    t1 = Nx.reshape(data, {1, n, dims})
    t2 = Nx.reshape(data, {n, 1, dims})

    t1 - t2
    |> Nx.power(2)
    |> Nx.sum(axes: [2])
    |> Nx.power(0.5)
    |> Nx.multiply(-1)
    |> Mlixir.fill_diagonal(@self_preference)
  end
end
