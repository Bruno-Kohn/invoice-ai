"""Pydantic schemas for Brazilian receipt/NF parsing (Nota Fiscal / Cupom Fiscal)."""

from pydantic import BaseModel
from typing import Optional


class ItemNF(BaseModel):
    """Um item de uma Nota Fiscal brasileira."""

    descricao: Optional[str] = None  # Descrição do produto
    quantidade: Optional[str] = None  # Quantidade (ex: "2")
    unidade: Optional[str] = None  # Unidade (UN, KG, LT)
    valor_unitario: Optional[str] = None  # Preço unitário
    valor_total: Optional[str] = None  # Valor total do item


class ImpostosNF(BaseModel):
    """Seção de impostos da NF."""

    icms: Optional[str] = None
    pis: Optional[str] = None
    cofins: Optional[str] = None
    valor_aproximado_tributos: Optional[str] = None


class NotaFiscal(BaseModel):
    """Nota Fiscal / Cupom Fiscal brasileiro estruturado."""

    # Identificação
    cnpj: Optional[str] = None
    razao_social: Optional[str] = None
    endereco: Optional[str] = None

    # Itens
    itens: list[ItemNF] = []

    # Valores
    subtotal: Optional[str] = None
    desconto: Optional[str] = None
    valor_total: Optional[str] = None
    forma_pagamento: Optional[str] = None  # Dinheiro, Cartão, PIX
    valor_pago: Optional[str] = None
    troco: Optional[str] = None

    # Impostos
    impostos: Optional[ImpostosNF] = None

    # Metadata
    cpf_consumidor: Optional[str] = None
    numero_nf: Optional[str] = None
    serie: Optional[str] = None
    data_emissao: Optional[str] = None
    chave_acesso: Optional[str] = None
