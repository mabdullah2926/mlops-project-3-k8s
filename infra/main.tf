resource "null_resource" "minikube_cluster" {
  provisioner "local-exec" {
    command = "minikube start --profile=${var.minikube_profile} --driver=${var.driver} --cpus=${var.cpus} --memory=${var.memory}"
  }
}
