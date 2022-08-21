defmodule Mlixir.NearestCentroid do
  @moduledoc """
  Model representing a nearest centroid classifier.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  """
  @impl true
  defn fit(x, y, opts \\ []) do
    {n, m} = Nx.shape(x)

    opts = keyword!(opts, [:n_classes])
    n_classes = get_n_classes(opts)

    classes = Nx.iota({n_classes, 1})
    masks = y
    |> Nx.broadcast({n_classes, n})
    |> Nx.equal(classes)

    class_counts = masks
    |> Nx.sum(axes: [1], keep_axes: true)
    |> Nx.broadcast({n_classes, m})

    masks = masks
    |> Nx.new_axis(2)
    |> Nx.broadcast({n_classes, n, m})

    inputs = Nx.broadcast(x, {n_classes, n, m})

    Nx.select(masks, inputs, 0)
    |> Nx.sum(axes: [1])
    |> Nx.divide(class_counts)
  end

  @doc """
  """
  @impl true
  defn predict(model, input) do
    {a, _} = Nx.shape(input)
    {n_classes, m} = Nx.shape(model)

    model = Nx.broadcast(model, {a, n_classes, m})
    input = Nx.new_axis(input, 1) |> Nx.broadcast({a, n_classes, m})

    model - input
    |> Nx.power(2)
    |> Nx.sum(axes: [2], keep_axes: true)
    |> Nx.power(0.5)
    |> Nx.argmin(axis: 1)
  end

  deftransformp get_n_classes(opts), do: Keyword.fetch!(opts, :n_classes)
end
