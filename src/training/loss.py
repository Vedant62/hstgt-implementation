import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskAwareLoss(nn.Module):
    """
    L_total = λ1 * L_MSE  +  λ2 * L_Dir

    L_MSE  = mean((y - ŷ)²)           ← penalises magnitude errors
    L_Dir  = mean(max(0, -y * ŷ))     ← penalises direction mistakes
    
    Optional Sharpe penalty:
    L_Sharpe = -Sharpe(ŷ)             ← reward risk-adjusted returns
    """

    def __init__(
        self,
        lambda_mse:    float = 1.0,
        lambda_dir:    float = 1.0,
        lambda_sharpe: float = 0.0,   # Set > 0 to include Sharpe term
    ):
        super().__init__()
        self.lambda_mse    = lambda_mse
        self.lambda_dir    = lambda_dir
        self.lambda_sharpe = lambda_sharpe

    def forward(
        self,
        y_pred: torch.Tensor,  # (N,)  predicted log returns
        y_true: torch.Tensor,  # (N,)  actual log returns
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, component_dict) for logging.
        """
        loss_mse = F.mse_loss(y_pred, y_true)

        loss_dir = torch.mean(torch.clamp(-y_true * y_pred, min=0.0))
 
        loss_sharpe = torch.tensor(0.0, device=y_pred.device)
        if self.lambda_sharpe > 0:
            # Simulated long/short: use pred sign as +1/-1 position
            portfolio_returns = y_true * torch.sign(y_pred)
            mean_ret = portfolio_returns.mean()
            std_ret  = portfolio_returns.std() + 1e-8
            sharpe   = mean_ret / std_ret
            loss_sharpe = -sharpe  # Minimise negative Sharpe

        # ── Total Loss ─────────────────────────────────────────────────────────
        total = (
            self.lambda_mse    * loss_mse +
            self.lambda_dir    * loss_dir +
            self.lambda_sharpe * loss_sharpe
        )

        components = {
            "loss_total":  total.item(),
            "loss_mse":    loss_mse.item(),
            "loss_dir":    loss_dir.item(),
            "loss_sharpe": loss_sharpe.item(),
        }
        return total, components


def directional_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> float:
    """Compute fraction of predictions with correct sign."""
    correct = (torch.sign(y_pred) == torch.sign(y_true)).float()
    return correct.mean().item()