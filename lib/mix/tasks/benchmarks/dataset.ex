defmodule Mix.Tasks.Benchmarks.Dataset do
  def prepare_mnist do
    Scidata.MNIST.download(
      transform_images: &transform_data/1,
      transform_labels: &transform_labels/1
    )
  end

  defp transform_data({binary, type, shape}) do
    binary
    |> Nx.from_binary(type)
    # |> Nx.reshape(shape)
    |> Nx.reshape({60_000, 784})
    |> Nx.slice([59_999, 0], [1, 784])
  end

  defp transform_labels({binary, type, _shape}) do
    binary
    |> Nx.from_binary(type)
    |> Nx.slice([59_999], [1])
  end
end
