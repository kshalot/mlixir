samples = 10
sample_length = 10

#TODO: Types
x64 = Nx.random_uniform({samples, sample_length})
y64 = Nx.random_uniform({samples})

x32 = Nx.random_uniform({samples, sample_length}, type: {:f, 32})
y32 = Nx.random_uniform({samples}, type: {:f, 32})

defmodule Benchmark.SGD do
  import Nx.Defn

  defn sgd(x, y) do
    Mlixir.SGD.fit(x, y)
    |> Mlixir.SGD.predict(x) #FIXME: Predicting the training set (shouldn't impact performance, but fix it later)
  end

  @defn_compiler EXLA
  def host(x, y), do: sgd(x, y)
end

Nx.Defn.aot(
  Benchmark.SGD.AOT,
  [
    {:sgd_64, &Benchmark.SGD.sgd/2, [x64, y64]},
    {:sgd_32, &Benchmark.SGD.sgd/2, [x32, y32]}
  ],
  compiler: EXLA
)

benches = %{
  "elixir f64" => fn -> Benchmark.SGD.sgd(x64, y64) end,
  "elixir f32" => fn -> Benchmark.SGD.sgd(x32, y32) end,
  "xla jit-cpu f64" => fn -> Benchmark.SGD.host(x64, y64) end,
  "xla jit-cpu f32" => fn -> Benchmark.SGD.host(x32, y32) end,
  "xla aot-cpu f64" => fn -> Benchmark.SGD.AOT.sgd_64(x64, y64) end,
  "xla aot-cpu f32" => fn -> Benchmark.SGD.AOT.sgd_32(x32, y32) end
}

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
