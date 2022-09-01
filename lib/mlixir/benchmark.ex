defmodule Mlixir.Benchmark do
  def affinity_propagation(n, k) do
    cuda_jit = EXLA.jit(&Mlixir.AffinityPropagation.fit/1)

    input = Nx.from_numpy("/net/scratch/people/plgkshalot/data/single/data_#{n}_#{k}.npy")

    benches = %{
      "#{n},#{k},f32" => {fn -> cuda_jit.(input) end, after_each: &Nx.backend_deallocate/1}
    }

    Benchee.run(
      benches,
      time: 1,
      formatters: [&Formatter.output/1]
    )

    input = Nx.from_numpy("/net/scratch/people/plgkshalot/data/data_#{n}_#{k}.npy")

    benches = %{
      "#{n},#{k},f64" => {fn -> cuda_jit.(input) end, after_each: &Nx.backend_deallocate/1}
    }

    Benchee.run(
      benches,
      time: 1,
      formatters: [&Formatter.output/1]
    )
  end

  def nearest_centroid_fit(n, k, l) do
    fit_cuda_jit = EXLA.jit(fn (x,y) -> Mlixir.NearestCentroid.fit(x, y, n_classes: l) end)

    x = Nx.from_numpy("/net/scratch/people/plgkshalot/data/single/data_#{n}_#{k}.npy")
    y = Nx.from_numpy("/net/scratch/people/plgkshalot/data/labels_#{n}_#{l}.npy")

    benches = %{
      "#{n},#{k},#{l},f32" => {fn -> fit_cuda_jit.(x, y) end, after_each: &Nx.backend_deallocate/1}
    }

    Benchee.run(
      benches,
      time: 1,
      formatters: [&Formatter.output/1]
    )

    x = Nx.from_numpy("/net/scratch/people/plgkshalot/data/data_#{n}_#{k}.npy")

    benches = %{
      "#{n},#{k},#{l},f64" => {fn -> fit_cuda_jit.(x, y) end, after_each: &Nx.backend_deallocate/1}
    }

    Benchee.run(
      benches,
      time: 1,
      formatters: [&Formatter.output/1]
    )
  end

  def nearest_centroid_predict(n, k, l) do
    predict_cuda_jit = EXLA.jit(fn (x,y) -> Mlixir.NearestCentroid.predict(x, y) end)

    m = Nx.from_numpy("/net/scratch/people/plgkshalot/data/single/model_#{l}_#{k}.npy")
    x = Nx.from_numpy("/net/scratch/people/plgkshalot/data/single/data_#{n}_#{k}.npy")

    benches = %{
      "#{n},#{k},#{l},f32" => {fn -> predict_cuda_jit.(m, x) end, after_each: &Nx.backend_deallocate/1}
    }

    Benchee.run(
      benches,
      time: 1,
      formatters: [&Formatter.output/1]
    )

    m = Nx.from_numpy("/net/scratch/people/plgkshalot/data/model_#{l}_#{k}.npy")
    x = Nx.from_numpy("/net/scratch/people/plgkshalot/data/data_#{n}_#{k}.npy")

    benches = %{
      "#{n},#{k},#{l},f64" => {fn -> predict_cuda_jit.(m, x) end, after_each: &Nx.backend_deallocate/1}
    }

    Benchee.run(
      benches,
      time: 1,
      formatters: [&Formatter.output/1]
    )
  end
end
