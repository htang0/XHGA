import time


def transform_model(feature_model_path, classifier_model_paths, classifier_out_dims):
    import ai_edge_torch
    import torch
    from model import FeatureExtractor, LinearClassifierAttn

    dataset = "ours"
    feature_model = FeatureExtractor(1, dataset)
    feature_model.load_state_dict(
        torch.load(feature_model_path, weights_only=True)["model"]
    )
    encoder = feature_model.encoder

    classifiers = []
    for p, d in zip(classifier_model_paths, classifier_out_dims):
        classifier = LinearClassifierAttn(d, dataset)
        classifier.load_state_dict(torch.load(p, weights_only=True)["model"])
        classifiers.append(classifier)

    encoder_in_samples = (torch.randn(1, 1, 300, 12), torch.randn(1, 3, 32, 72, 128))
    encoder = ai_edge_torch.convert(encoder.eval(), sample_args=encoder_in_samples)
    encoder.export("./save/litert_models/encoder.tflite")

    for i, classifier in enumerate(classifiers):
        classifier_in_samples = (
            torch.randn(1, 16, 97, 4),
            torch.randn(1, 512, 1, 3, 5),
        )
        classifier = ai_edge_torch.convert(
            classifier.eval(), sample_args=classifier_in_samples
        )
        classifier.export(f"./save/litert_models/classifier{i}.tflite")


def eval_latency(encoder_tf_path, classifier_tf_paths, num_runs=20, warmup_runs=10):
    from ai_edge_litert.interpreter import Interpreter
    import numpy as np

    try:
        print(f"Loading encoder: {encoder_tf_path}")
        encoder_interpreter = Interpreter(model_path=encoder_tf_path, num_threads=4)
        print(f"Loading {len(classifier_tf_paths)} classifiers...")
        classifier_interpreters = [
            Interpreter(model_path=path) for path in classifier_tf_paths
        ]

        encoder_interpreter.allocate_tensors()
        for interp in classifier_interpreters:
            interp.allocate_tensors()

        encoder_input_details = encoder_interpreter.get_input_details()
        encoder_output_details = encoder_interpreter.get_output_details()
        classifier_input_details_list = [
            interp.get_input_details() for interp in classifier_interpreters
        ]

        encoder_output_index_1 = encoder_output_details[0]["index"]
        encoder_output_index_2 = encoder_output_details[1]["index"]

        classifier_input_indices_list = []
        for i, c_input_details in enumerate(classifier_input_details_list):
            c_input_index_1 = c_input_details[0]["index"]
            c_input_index_2 = c_input_details[1]["index"]
            classifier_input_indices_list.append((c_input_index_1, c_input_index_2))

        # Use details from the loaded model to ensure correct shape and dtype
        input_data_enc_1 = np.random.randn(*encoder_input_details[0]["shape"]).astype(
            encoder_input_details[0]["dtype"]
        )
        input_data_enc_2 = np.random.randn(*encoder_input_details[1]["shape"]).astype(
            encoder_input_details[1]["dtype"]
        )

        print(f"Performing {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs):
            encoder_interpreter.set_tensor(
                encoder_input_details[0]["index"], input_data_enc_1
            )
            encoder_interpreter.set_tensor(
                encoder_input_details[1]["index"], input_data_enc_2
            )
            encoder_interpreter.invoke()
            # Get Encoder Outputs
            encoder_output_data_1 = encoder_interpreter.get_tensor(
                encoder_output_index_1
            )
            encoder_output_data_2 = encoder_interpreter.get_tensor(
                encoder_output_index_2
            )

            # Run Classifiers
            for i, interp in enumerate(classifier_interpreters):
                classifier_input_idx_1, classifier_input_idx_2 = (
                    classifier_input_indices_list[i]
                )
                # Set Classifier Inputs
                interp.set_tensor(classifier_input_idx_2, encoder_output_data_1)
                interp.set_tensor(classifier_input_idx_1, encoder_output_data_2)
                # Invoke Classifier
                interp.invoke()

        # Benchmarking Loop
        print(f"Running benchmark ({num_runs} runs)...")
        latencies = []
        encoder_latencies = []
        for _ in range(num_runs):
            start_time = time.perf_counter()

            encoder_interpreter.set_tensor(
                encoder_input_details[0]["index"], input_data_enc_1
            )
            encoder_interpreter.set_tensor(
                encoder_input_details[1]["index"], input_data_enc_2
            )

            encoder_interpreter.invoke()

            encoder_output_data_1 = encoder_interpreter.get_tensor(
                encoder_output_index_1
            )
            encoder_output_data_2 = encoder_interpreter.get_tensor(
                encoder_output_index_2
            )

            encoder_latencies.append((time.perf_counter() - start_time) * 1000)

            for i, interp in enumerate(classifier_interpreters):
                classifier_input_idx_1, classifier_input_idx_2 = (
                    classifier_input_indices_list[i]
                )

                interp.set_tensor(classifier_input_idx_2, encoder_output_data_1)
                interp.set_tensor(classifier_input_idx_1, encoder_output_data_2)
                interp.invoke()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        latencies_ms = np.array(latencies)
        avg_latency = np.mean(latencies_ms)
        std_latency = np.std(latencies_ms)
        p95_latency = np.percentile(latencies_ms, 95)

        encoder_latencies_ms = np.array(encoder_latencies)
        avg_encoder_latency = np.mean(encoder_latencies_ms)
        std_encoder_latency = np.std(encoder_latencies_ms)

        print("--- Benchmarking Results ---")
        print(f"Number of runs: {num_runs}")
        print(f"Average Latency:  {avg_latency:.3f} ms")
        print(f"Std Dev Latency:  {std_latency:.3f} ms")
        print(f"95th Percentile:  {p95_latency:.3f} ms")
        print()
        print(f"Encoder Average Latency:  {avg_encoder_latency:.3f} ms")
        print(f"Encoder Std Dev Latency:  {std_encoder_latency:.3f} ms")
        print("----------------------------")

        results = {
            "avg_ms": avg_latency,
            "std_ms": std_latency,
            "p95_ms": p95_latency,
            "num_runs": num_runs,
            "avg_encoder_ms": avg_encoder_latency,
            "std_encoder_ms": std_encoder_latency,
            "num_classifiers": len(classifier_interpreters),
        }
        return results

    except Exception as e:
        print(f"An error occurred during latency evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None
